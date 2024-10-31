import os
import argparse
import multiprocessing
import shutil
from multiprocessing import Pool
from typing import Optional
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.configuration import default_num_processes
import numpy as np
import mlflow


def split_4d_nifti(filename, output_folder):
    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = os.path.basename(filename)
    if dim == 3:
        shutil.copy(filename, join(output_folder, file_base[:-7] + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, filename))
    else:
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection()).reshape(4,4)
        # now modify these to remove the fourth dimension
        spacing = tuple(list(spacing[:-1]))
        origin = tuple(list(origin[:-1]))
        direction = tuple(direction[:-1, :-1].reshape(-1))
        for i, t in enumerate(range(img_npy.shape[0])):
            img = img_npy[t]
            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(spacing)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(img_itk_new, join(output_folder, file_base[:-7] + "_%04.0d.nii.gz" % i))


def convert_msd_dataset(source_folder: str, output_folder: str, num_processes: int = default_num_processes) -> None:

    # Source folder
    if source_folder.endswith('/') or source_folder.endswith('\\'):
        source_folder = source_folder[:-1]
    labelsTr = join(source_folder, 'labelsTr')
    imagesTs = join(source_folder, 'imagesTs')
    imagesTr = join(source_folder, 'imagesTr')
    assert isdir(labelsTr), f"labelsTr subfolder missing in source folder"
    assert isdir(imagesTs), f"imagesTs subfolder missing in source folder"
    assert isdir(imagesTr), f"imagesTr subfolder missing in source folder"
    dataset_json = join(source_folder, 'dataset.json')
    assert isfile(dataset_json), f"dataset.json missing in source_folder"

    # Target folder
    target_folder = output_folder
    target_imagesTr = join(target_folder, 'imagesTr')
    target_imagesTs = join(target_folder, 'imagesTs')
    target_labelsTr = join(target_folder, 'labelsTr')
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []

        # convert 4d train images
        source_images = [i for i in subfiles(imagesTr, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [join(imagesTr, i) for i in source_images]

        results.append(
            p.starmap_async(
                split_4d_nifti, zip(source_images, [target_imagesTr] * len(source_images))
            )
        )

        # convert 4d test images
        source_images = [i for i in subfiles(imagesTs, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [join(imagesTs, i) for i in source_images]

        results.append(
            p.starmap_async(
                split_4d_nifti, zip(source_images, [target_imagesTs] * len(source_images))
            )
        )

        # copy segmentations
        source_images = [i for i in subfiles(labelsTr, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        for s in source_images:
            shutil.copy(join(labelsTr, s), join(target_labelsTr, s))

        [i.get() for i in results]

    dataset_json = load_json(dataset_json)
    print(str(dataset_json))
    dataset_json['labels'] = {j: int(i) for i, j in dataset_json['labels'].items()}
    dataset_json['file_ending'] = ".nii.gz"
    dataset_json["channel_names"] = dataset_json["modality"]
    try:
        del dataset_json["modality"]
    except:
        print("modality not found")
    try:
        del dataset_json["training"]
    except:
        print("training not found")
    try:
        del dataset_json["test"]
    except:
        print("test not found")
    try:
        del dataset_json["tensorImageSize"]
    except:
        print("tensorImageSize not found")
    try:
        del dataset_json["numTest"]
    except:
        print("numTest not found")
    print(str(dataset_json))
    save_json(dataset_json, join(target_folder, 'dataset.json'), sort_keys=False)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='Downloaded and extracted MSD dataset folder.')
    parser.add_argument('-o', type=str, required=True,
                        help='Converted output dataset folder')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'Number of processes used. Default: {default_num_processes}')
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print('args:'+str(args))
    print('input dir contents: '+str(os.listdir(args.i)))
    convert_msd_dataset(args.i, args.o, args.np)

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
