import os
from style_transfer import run_style_transfer
from PIL import Image

iterations = set((10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000))
style_weights = [1e-2, 1e-1, 1, 10, 100, 1000]
styles = [
  ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
  ['block1_conv1'],
  ['block2_conv1'],
  ['block3_conv1'],
  ['block4_conv1'],
  ['block5_conv1']
]

images = [
  ("luna.jpg", "escher.jpg"),
  ("fireweed.jpg", "gala.jpg"),
  ("flower.jpg", "The_Great_Wave_off_Kanagawa.jpg")
]

def main():
  # Quick sanity check
  for content_image, style_image in images:
    if not os.path.isfile(os.path.join("images", content_image)):
      print(f"Could not find {content_image}")
      exit()
    if not os.path.isfile(os.path.join("images", style_image)):
      print(f"Could not find {style_image}")
      exit()

  for content_image, style_image in images:
    content_path = os.path.join("images", content_image)
    content_prefix = os.path.splitext(content_image)[0]
    style_path = os.path.join("images", style_image)
    style_prefix = os.path.splitext(style_image)[0]
    for style_layers in styles:
      for style_weight in style_weights:
        output_path = f"param_sweep/{content_prefix}_{style_prefix}/{''.join(style_layers)}/weight_{style_weight}/"
        if not os.path.exists(output_path):
          os.makedirs(output_path)
        # run style transfer...
        best, best_loss, images = run_style_transfer(content_path, style_path,
          style_weight=style_weight,
          style_layers=style_layers,
          num_iterations=1000,
          save_iters=iterations
        )
        for num_iterations, image in images.items():
          im = Image.fromarray(image)
          im.save(os.path.join(output_path, f"iter_{num_iterations}.png"), "PNG")


if __name__ == "__main__":
  main()
