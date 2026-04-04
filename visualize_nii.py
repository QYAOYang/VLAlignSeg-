"""
可视化 NIfTI 文件
"""
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

# 读取 NIfTI 文件
def load_nii_file(file_path):
    """
    加载 NIfTI 文件
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

# 可视化图像
def visualize_images(image, label, save_path=None):
    """
    可视化图像和标签
    """
    # 选择中间切片进行可视化
    if len(image.shape) == 3:
        mid_slice = image.shape[2] // 2
        image_slice = image[:, :, 1]
        label_slice = label[:, :, 1]
    else:
        image_slice = image
        label_slice = label
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 显示图像
    im1 = ax1.imshow(image_slice, cmap='gray')
    ax1.set_title('Image')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    # 显示标签
    im2 = ax2.imshow(label_slice, cmap='gray')
    ax2.set_title('Label')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    # 显示叠加效果
    overlay = image_slice.copy()
    overlay[label_slice > 0.5] = overlay[label_slice > 0.5] * 0.5 + 0.5
    im3 = ax3.imshow(overlay, cmap='gray')
    ax3.set_title('Overlay')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    plt.show()

# 主函数
def main():
    # 输入文件路径
    image_file = '/home/qhp/MLstudy/ssl_agent_exp/CoW/data/CHAOST2/chaos_MR_T2_normalized/superpix-MIDDLE_1.nii.gz'
    label_file = '/home/qhp/MLstudy/ssl_agent_exp/CoW/data/CHAOST2/chaos_MR_T2_normalized/superpix-MIDDLE_2.nii.gz'
    
    # 输出文件路径
    output_dir = '/home/qhp/MLstudy/ssl_agent_exp'
    visualization_file = os.path.join(output_dir, 'image_label_visualization.png')
    
    # 加载数据
    print(f'Loading image file: {image_file}')
    image_data, image_affine = load_nii_file(image_file)
    print(f'Image shape: {image_data.shape}')
    
    print(f'Loading label file: {label_file}')
    label_data, label_affine = load_nii_file(label_file)
    print(f'Label shape: {label_data.shape}')
    
    # 可视化结果
    print(f'Visualizing results...')
    visualize_images(image_data, label_data, visualization_file)
    print(f'Visualization saved to: {visualization_file}')
    
    print('Visualization completed successfully!')

if __name__ == '__main__':
    main()
