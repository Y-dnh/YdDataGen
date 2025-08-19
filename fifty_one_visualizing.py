import fiftyone as fo

def launch_fiftyone_bbox_visualizing(config):
    """
    Launches FiftyOne to visualize bounding boxes on a COCO-annotated dataset.

    Args:
        config: A configuration object containing paths to the dataset and annotation files.
    """

    # data_path= config.dataset_dir / 'test'
    # labels_path= config.dataset_dir / "annotations/instances_test.json"

    data_path= "/dataset/data/fOJUh03ttqY"
    labels_path= "C:/YtDataGen/dataset/labels_final1.json"

    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
    )

    session = fo.launch_app(dataset, port=5151)
    session.wait()

if __name__ == '__main__':
    from config import create_config

    config = create_config()
    launch_fiftyone_bbox_visualizing(config)


# FilterLabels, detection, ..., True, False
"""
🔎 Example of using a custom bounding box area filter inside the GUI (Filter > detections):

This Mongo-style expression filters out detections with `bbox_area` between 0.2% and 0.25% 
of the image area:

{
  "$and": [
    {
      "$gte": [
        "$$this.bbox_area",
        0.002
      ]
    },
    {
      "$lte": [
        "$$this.bbox_area",
        0.0025
      ]
    }
  ]
}

Note:
- `bbox_area` is not automatically present.
- You’ll need to manually compute and add it (e.g., via a stage or a script).
"""