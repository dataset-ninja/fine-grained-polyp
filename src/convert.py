import csv
import os
import shutil
from urllib.parse import unquote, urlparse

import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    train_data_path = "/home/alex/DATASETS/TODO/polyp_ash/m_train/images"
    val_data_path = "/home/alex/DATASETS/TODO/polyp_ash/m_valid/images"
    test_data_path = "/home/alex/DATASETS/TODO/polyp_ash/m_test/images"

    train_masks_path = "/home/alex/DATASETS/TODO/polyp_ash/m_train/masks"
    train_tags_path = "/home/alex/DATASETS/TODO/polyp_ash/m_train/train.csv"

    val_masks_path = "/home/alex/DATASETS/TODO/polyp_ash/m_valid/masks"
    val_tags_path = "/home/alex/DATASETS/TODO/polyp_ash/m_valid/valid.csv"

    test_masks_path = "/home/alex/DATASETS/TODO/polyp_ash/m_test/masks"
    test_tags_path = "/home/alex/DATASETS/TODO/polyp_ash/m_test/test.csv"

    batch_size = 30

    ds_name_to_split = {
        "train": (train_data_path, train_masks_path, train_tags_path),
        "val": (val_data_path, val_masks_path, val_tags_path),
        "test": (test_data_path, test_masks_path, test_tags_path),
    }

    def create_ann(image_path):
        labels = []

        tags_data = image_name_to_tags[get_file_name(image_path)]
        histologia = sly.Tag(histologia_meta, value=tags_data[0])
        cls_meta = cls_to_meta.get(tags_data[1])
        cls_tag = sly.Tag(cls_meta)

        mask_path = os.path.join(masks_path, get_file_name_with_ext(image_path))
        mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
        img_height = mask_np.shape[0]
        img_wight = mask_np.shape[1]
        obj_mask = mask_np == 255
        ret, curr_mask = connectedComponents(obj_mask.astype("uint8"), connectivity=8)
        for i in range(1, ret):
            obj_mask = curr_mask == i
            curr_bitmap = sly.Bitmap(obj_mask)
            curr_label = sly.Label(curr_bitmap, obj_class)
            labels.append(curr_label)

        return sly.Annotation(
            img_size=(img_height, img_wight), labels=labels, img_tags=[histologia, cls_tag]
        )

    obj_class = sly.ObjClass("polyp", sly.Bitmap)

    histologia_meta = sly.TagMeta("histologia", sly.TagValueType.ANY_STRING)

    ad_meta = sly.TagMeta("ad", sly.TagValueType.NONE)
    ass_meta = sly.TagMeta("ass", sly.TagValueType.NONE)
    hp_meta = sly.TagMeta("hp", sly.TagValueType.NONE)

    cls_to_meta = {"AD": ad_meta, "ASS": ass_meta, "HP": hp_meta}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class], tag_metas=[histologia_meta, ad_meta, ass_meta, hp_meta]
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, ds_data in ds_name_to_split.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_path, masks_path, tags_path = ds_data

        image_name_to_tags = {}
        with open(tags_path, "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                image_name_to_tags[row[0]] = row[1:]

        images_names = os.listdir(images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
