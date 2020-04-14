---
layout: post
description: Splitting a folder of photos in a random subset of training and testing photos is not readily available as a program. This little program provides a solution.
categories: [Python]
comments: true
hide: true
---

# Custom Data Splitter Function in Python

Ths month I started with the excellent Deep Learning course by [fast.ai](https://www.fast.ai) and I just finished the first lesson. At the end of the lesson Jeremy, the teacher of the course, assigned a homework assignment to create an image classifier using our own data. After downloading my own dataset I was facing the problem that I wanted to split this dataset into a training and test set using seperate directories but I could not easily find such a program online. Therefore, in this blog post, I will explain how to write such a program and how to make this program easily installable and useable for anyone. I will discuss how to properly handle arguments for command line programs and how to make your software available to download from `pip`.

## Motivation

Most of these image classifier tutorials use a dataset that correctly classifies most of the images using state-of-the-art image classifier models. However, I was curious how these so-called transfer learning techniques would work on a rather challenging dataset. Using a google image downloader, I obtained a dataset with approximately 1,000 Airbus A320 planes and 1,000 Boeing 737s. For a layman, these planes are identical but there are small differences.

The problem I was facing is that created a directory `plane_data` and two subdirectories `airbus` and `boeing` containing the respective image files but I wanted to split this dataset into a training and test dataset. While it is relatively straight-forward to split a data file, e.g., CSV files containing one observation per row, splitting directories containing image files is less straightforward. Hence, I decided to write a little terminal program in python to do the manual work for me.

## The Program

The goal of the program is split a directory of images and split it into a training and test directory. 

Let's call our directory with images `data` with classes **boeing** and **airbus**. The two most used ways of saving images beloning to a certain class is to put them in a seperate subdirectory, e.g., `data/boeing` or to put the class name in the name of the image file, e.g., `boeing_img1.jpeg`. As of now, I only implemented the former case but the second case is typically implemented using class detection with *regex*.

The source code of the program is given below. I think the code is pretty self-explanatory but the idea is to 1) identify the subdirectories containing all classes, 2) create a `data/train` and `data/test` subdirectory and 3) loop over all classes and split the image files in a train and test selection and move the images to the respective directories, e.g., `data/train/boeing` and `data/test/boeing`. Note that the `folder` variable is the location of the images and the `train` variable is the percentage of images that we want to keep for training, which is defined as a program option.

```python
def data_splitter(folder, train):
  entries = [item for item in listdir(folder) if not item.startswith('.')]

  # Get all folders and files
  dirs = [d for d in entries if isdir(join(folder, d))]
  files = [f for f in entries if isfile(join(folder, f))]

  if dirs and files:
    print("Folder should contain either files or folders but not both.")
    sys.exit()

  mkdir(join(folder, 'train'))
  mkdir(join(folder, 'test'))

  if files:
    dirs = [""]

  for directory in dirs:
    # Items belonging to the current class
    items = [item for item in listdir(join(folder, directory)) if not item.startswith('.') and isfile(join(folder, directory, item))]

    if directory != "":
      mkdir(join(folder, 'train', directory))
      mkdir(join(folder, 'test', directory))

    # Shuffle and split dataset according to fractions
    random.shuffle(items)
    train_sel = int(len(items) * (train / 100))
    train_entries = items[0:train_sel]
    test_entries = items[train_sel:]

    for train_file in train_entries:
      shutil.move(join(folder, directory, train_file), join(folder, 'train', directory, train_file))
    for test_file in test_entries:
      shutil.move(join(folder, directory, test_file), join(folder, 'test', directory, test_file))

    if directory != "":
      rmdir(join(folder, directory))
```

## Handling Arguments

The program needs a folder to split up and an optional percentage of image files to put in the training set. While python offers an in-built argument parser in the `argparser` module, the `click` module has my preference due to its easy of use.

Adding arguments and options is as simple as adding
```python
@click.command()
@click.argument('folder')
@click.option('--train', default=80, help="Percentage of files for train set")
```

In front of our `data_splitter(folder, train)` program. A `click.option` allows us to define a default value in case the user does not define the option and a little helper text when the user calls the `--help` option.

## Deploying on pip

Lastly, we want to make it dead-easy to install and use our little command-line program. The most popular method to distribute software in python is to use `pip`. To make our module suitable for `pip`, we need to add two files to our module in the main directory: a `setup.py` and `setup.cfg` file. An example can be found [here](https://github.com/pypa/sampleproject).

We further add the following to the `data_splitter.py` code to make the program easily calleable from pip as we will see in the next section.

```python
if __name__ == '__main__':
  try:
    data_splitter()
  except FileNotFoundError as fnf_error:
    print(fnf_error)
```

The try-except block prevents us from having an ugly and long error callback in case we want to split a non-existing directory.

## Using data_splitter

Now that we defined the program, handled program arguments and made our program deployable on `pip` we can start using it.

The program can be found in the directory https://github.com/markkvdb/data-splitter and can be installed as

```bash
pip install git+https://github.com/markkvdb/data-splitter.git#egg=data-splitter
```

The installer will place the program location in the `$PATH` variable of your system.

After only adding three lines of code using the `click` module, we now have a very neat interface when using the program in the command-line. Calling `data_splitter --help` will show how to use it

```bash
Usage: data_splitter [OPTIONS] FOLDER

Options:
  --train INTEGER  Percentage of files for train set.
  --help           Show this message and exit.
```

## Improvements

The simple command-line can be improved in various ways:

- Implement class recognition using image file names with regex;
- Splitting is performed by randomly assigning images to the test or training set. This is not always appriorate if some images belong to the same individual and images are no longer independent.
