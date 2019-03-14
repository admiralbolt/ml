import matplotlib.pyplot as plt
import matplotlib as mpl
import param_sweep
import image_utils
import os

plt.figure(figsize=(10, 10))

for content_image, style_image in param_sweep.images:
  plt.clf()
  i = 0
  # f, axarr = plt.subplots(len(param_sweep.styles), len(param_sweep.style_weights))
  content_path = os.path.join("images", content_image)
  content_prefix = os.path.splitext(content_image)[0]
  style_path = os.path.join("images", style_image)
  style_prefix = os.path.splitext(style_image)[0]
  for style_layers in param_sweep.styles:
    for style_weight in param_sweep.style_weights:
      output_path = f"param_sweep/{content_prefix}_{style_prefix}/{''.join(style_layers)}/weight_{style_weight}/iter_1000.png"
      # axarr[i, j].imshow(image_utils.load_image(output_path))
      plt.subplot(len(param_sweep.styles), len(param_sweep.style_weights), i + 1)
      image_utils.show_image(image_utils.load_image(output_path), f"{''.join(style_layers)}-weight_{style_weight}")
      i += 1
  plt.show()
