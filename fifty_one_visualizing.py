import fiftyone as fo

def launch_fiftyone_bbox_visualizing():
    data_path= "C:/YtDataGen/dataset/data/fOJUh03ttqY"
    labels_path= "C:/YtDataGen/dataset/fOJUh03ttqY_annotations.json"

    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
    )

    session = fo.launch_app(dataset, port=5151)
    session.wait()

if __name__ == '__main__':
    launch_fiftyone_bbox_visualizing()


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