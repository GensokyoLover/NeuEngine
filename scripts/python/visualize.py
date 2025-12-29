from utils import * 
from impostor import *
def show_5_images_click_single_red_block(
    sample,
    impostor,
    window_name: str = "Click left image: single 8x8 red block (ESC/q quit)",
    block: int = 8,
) -> Optional[Tuple[int, int]]:
    roughness = torch.round(torch.arange(0.0, 1.0 + 1e-9, 0.01) * 100) / 100
    theta95_deg = ggx_theta99_from_roughness(roughness, alpha_is_roughness_sq=True, degrees=True).cuda()
    print(theta95_deg)
    img0_base = to_uint8_rgb(sample["AccumulatePassoutput"].cpu().numpy()).copy()
    img0_work = img0_base.copy()

    imgs4 = [
        np.zeros((512, 512, 3), dtype=np.uint8),
        np.zeros((512, 512, 3), dtype=np.uint8),
        np.zeros((512, 512, 3), dtype=np.uint8),
        np.zeros((512, 512, 3), dtype=np.uint8),
    ]

    reflect_uvi = sample["reflect_uvi"].reshape(512, 512, 3, 3)  # ✅ 提前算一次

    state = {
        "dirty": True,
        "last_click": None,   # (x,y)
        "prev_bbox": None,
        "roughness": 0.0,
    }

    
    # ✅ 把“点击后更新”的逻辑抽出来：点击和滑块变化都调用它
    def update_from_xy(x: int, y: int):
        nonlocal imgs4

        # only accept clicks in left image
        if not (0 <= x < 512 and 0 <= y < 512):
            return

        H, W, _ = img0_work.shape

        # 1) undo previous red block
        if state["prev_bbox"] is not None:
            px0, px1, py0, py1 = state["prev_bbox"]
            img0_work[py0:py1, px0:px1, :] = img0_base[py0:py1, px0:px1, :]

        # 2) paint new red block
        x0, x1, y0, y1 = compute_block_bbox(x, y, block, W, H)
        img0_work[y0:y1, x0:x1, :] = np.array([255, 0, 0], dtype=np.uint8)

        state["prev_bbox"] = (x0, x1, y0, y1)
        ### visualize which inside cone ? 
        # inside_mask = inside_cone(sample,impostor)
        # pyexr.write("inside_cone.exr",inside_mask.float().reshape(512,512,1).cpu().numpy())

        view = sample["reflect_uvi"].reshape(512, 512, 3, 3) 
        refer_camera = view[y,x,:,2]
        triCameraIdx = refer_camera.repeat(4096,1).long()
        reflect_pos =sample["refelctPosL"].reshape(512, 512, 3)[y,x].unsqueeze(0).repeat(64 * 64 ,1)
        roughness_level = int(state["roughness"] * 100)
        half_angle = theta95_deg[roughness_level]

        reflect_dir =sample_cone_directions_grid_torch(sample["reflectDirL"].reshape(512, 512, 3)[y,x].unsqueeze(0), half_angle, H=64, W=64, angle_is_full=False, flatten_hw=False)[0]
        dirs = reflect_dir.detach().cpu().numpy().astype(np.float32)  # (256,256,3)
        pts = dirs.reshape(-1, 3)  # point cloud on unit sphere
        new_reflect_uvi,final_pos = chunk_process(impostor,reflect_dir.reshape(-1,3),reflect_pos, W=W, H=H, chunk=512*512//8)
        proj_uv = []
        proj_viewIdx = triCameraIdx  # (B,3)

        for i in range(3):
            uv_i = project_world_to_view_uv(
                impostor,
                final_pos,                     # (B,3)
                triCameraIdx[:, i]
            )   # (B,2)

            proj_uv.append(uv_i)

        proj_uv = torch.stack(proj_uv, dim=1)   # (B,3,2)

        proj_uvi = torch.cat([proj_uv,proj_viewIdx.unsqueeze(-1).float()],dim=-1)
        new_reflect_uvi = proj_uvi.reshape(64,64,3,3)
        view = new_reflect_uvi
        cols = ((pts * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

        for i in range(3):
            view_id = int(view[0,0,i, 2].item()) if hasattr(view[0,0,i, 2], "item") else int(view[0,0,i, 2])
                    # 这里按你的写法：H,W,V
            view_image = impostor.texDict["emission"][:, :, view_id].clone() * 1000
            for ix in range(64):
                for iy in range(64):
                    

                    # 你的 uv -> pixel
                    xy = (view[ix,iy,i, :2] * 0.5 + 0.5) * (impostor.texDict["emission"].shape[1] - 1)

                    # 转成 int（兼容 torch / numpy）
                    xi = int(xy[0].item()) if hasattr(xy[0], "item") else int(xy[0])
                    yi = int(xy[1].item()) if hasattr(xy[1], "item") else int(xy[1])

                    xi = max(0, min(xi, impostor.texDict["emission"].shape[1] - 1))
                    yi = max(0, min(yi, impostor.texDict["emission"].shape[0] - 1))

                    # 标红/标记区域（这里你是写 0.99）
                    y0p = max(yi - 4, 0)
                    y1p = min(yi + 4, impostor.texDict["emission"].shape[0])
                    x0p = max(xi - 4, 0)
                    x1p = min(xi + 4, impostor.texDict["emission"].shape[1])
                    view_image[y0p:y1p, x0p:x1p] = 1000

                    # ✅ 让 roughness 影响你的显示/结果（示例：简单缩放，不想要就删掉）
                    # view_image = (view_image * (0.2 + 0.8 * roughness)).clamp(0, 1)
                print(ix)
            imgs4[i] = to_uint8_rgb(view_image.detach().cpu().numpy())
            

        imgs4[3] = np.zeros((512, 512, 3), dtype=np.uint8)

        state["dirty"] = True

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state["last_click"] = (x, y)
        update_from_xy(x, y)

    # ✅ 滑块变化：如果之前点过，就用 last_click 重新跑一遍 update_from_xy
    def on_roughness_trackbar(pos_int: int):
        state["roughness"] = pos_int / 100.0
        if state["last_click"] is not None:
            x, y = state["last_click"]
            update_from_xy(x, y)  # ✅ 相当于“重新触发 on_mouse 的逻辑”

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse, {"sample": sample, "impostor": impostor})

    # 0..100 -> 0.00..1.00 step=0.01
    cv2.createTrackbar(
        "roughness (x0.01)",
        window_name,
        int(round(state["roughness"] * 100)),
        100,
        on_roughness_trackbar
    )

    canvas_bgr = None
    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if state["dirty"]:
            canvas_rgb = compose_canvas(img0_work, imgs4)
            canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
            state["dirty"] = False

        cv2.imshow(window_name, canvas_bgr)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    return state["last_click"]