from collators.qwen2_vision_process import set_max_pixels, fetch_image

# 调用其他函数
image_info = {
    "image_url": "/group/40077/chaxjli/Retrieve/LamRA/white.jpg",
    "min_pixels": 4 * 28 * 28,
    "max_pixels": 500 * 28 * 28,
}
resized_image = fetch_image(image_info)
print("Resized image:", resized_image)

# 设置 MAX_PIXELS 的值
set_max_pixels(500 * 28 * 28)

# 调用其他函数
image_info = {
    "image_url": "/group/40077/chaxjli/Retrieve/LamRA/white.jpg",
    "min_pixels": 4 * 28 * 28,
    "max_pixels": 500 * 28 * 28,
}
resized_image = fetch_image(image_info)
print("Resized image:", resized_image)