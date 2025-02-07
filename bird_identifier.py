from duckduckgo_search import DDGS
from fastdownload import download_url
from fastai.data.all import *
from fastai.vision.all import *

ddg = DDGS()
searches = 'forest','animal', 'bird'
path = Path('bird_or_not')

def search_images(keywords):
  bird_images = ddg.images(
      keywords=keywords,
      max_results=100,
  )
  return [img['image'] for img in bird_images]

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    resize_images(path/o, max_size=400, dest=path/o)
  
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    # The inputs to our model are images, and the outputs are categories (in this case, "bird" or "forest").
    blocks=(ImageBlock, CategoryBlock),
    # To find all the inputs to our model, run the get_image_files function (which returns a list of all image files in a path).
    get_items=get_image_files,
    # Split the data into training and validation sets randomly, using 20% of the data for the validation set.
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    # The labels (y values) is the name of the parent of each file (i.e. the name of the folder they're in, which will be bird or forest).
    get_y=parent_label,
    # Before training, resize each image to 192x192 pixels by "squishing" it (as opposed to cropping it).
    item_tfms=[Resize(128, method=ResizeMethod.Crop)]
).dataloaders(path)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)
learn.recorder.plot_loss()

is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))

print(f"This is: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")