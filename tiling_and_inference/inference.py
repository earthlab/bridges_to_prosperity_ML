#!/usr/bin/env python
import argparse
import geopandas as gpd
from fastai.vision import *
import boto3
import time


def main(input_file: str, model_path: str, shape_path: str, tiling_dir: str, input_tiff_path: str,
         location_name: str, output_file: str):
    job_id = os.path.basename(output_file).split('.json')[0]
    work_dir = os.path.dirname(output_file)

    t1 = time.time()
    # pkl file path
    learn_infer = load_learner(path=os.path.dirname(model_path), file=os.path.basename(model_path))

    # shape file path
    test_gdf = gpd.read_file(shape_path)

    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    np.random.seed(42)

    # tiling dir path
    test_data = ImageList.from_folder(tiling_dir).split_none().label_empty().transform(
        tfms, size=224, tfm_y=False).databunch().normalize(imagenet_stats)
    test_data.train_dl.new(shuffle=False)
    val_dict = {1: 'yes', 0: 'no'}
    geoms = []

    # input tiff file
    sent_indx = str(input_tiff_path.split('.')[0][-3:])
    with open(input_file, 'r') as f:
        file_data = json.load(f)
    ls_names = file_data['tiling_files']
    print(f"{time.time() - t1}s setup time")
    # unique name for progress csv
    csv_path = os.path.join(work_dir, f'{location_name}_{sent_indx}_Inference_Progress_{job_id}.csv')
    log_path = os.path.join(work_dir, f'{location_name}_{sent_indx}_Inference_Progress_{job_id}.log')
    with open(csv_path, 'w+') as f:
        f.write('Index,Filename,Geometry,Predicted_Label,Predicted_Value,fl_value\n')
    with open(log_path, 'w+') as f:
        f.write(f'Length of tiling files for job {len(ls_names)}\n')
    tg = np.array([os.path.basename(i) for i in test_gdf['name_shp']])
    # tiling file paths
    for i, y in enumerate(ls_names):
        t0 = time.time()
        diff1 = None
        diff2 = None
        try:
            t1 = time.time()
            testt = np.where(tg == os.path.basename(y)[:-4] + '.shp')
            diff1 = time.time() - t1
            try:
                if testt[0][0] != 0:
                    t1 = time.time()
                    im = test_data.train_ds[i][0]
                    prediction = learn_infer.predict(im)

                    pred_val = prediction[1].data.item()
                    pred_label = val_dict[pred_val]
                    fl_val = prediction[2].data[pred_val].item()

                    geom_bounds = list(list(test_gdf.iterrows())[testt[0][0]][1].geometry.boundary.coords)
                    geoms.append((y, geom_bounds, pred_label, pred_val, fl_val))
                    outline = f'{i},{y},{geom_bounds},{pred_label},{pred_val},{fl_val}'
                    with open(csv_path, 'a') as f:
                        f.write(outline + '\n')
                    diff2 = time.time() - t1
                else:
                    outline = 'First index'

                del im, prediction, pred_val, pred_label, fl_val, testt

            except Exception as e:
                outline = str(e)
        except:
            outline = 'failure'

        if diff1 is not None:
            outline += f" diff1 {diff1}s"
        if diff2 is not None:
            outline += f" diff2 {diff2}s"
        outline += f' {time.time() - t0}s'

        with open(log_path, 'a') as f:
            outline += '\n'
            f.write(outline)

    with open(output_file, 'w+') as f:
        json.dump({'geoms': geoms}, f, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to json file with tiling file paths')
    parser.add_argument('--model_path', type=str, required=True, help='Path to ML model')
    parser.add_argument('--shape_path', type=str, required=True, help='Path to input shp file')
    parser.add_argument('--tiling_dir', type=str, required=True, help='Path to tiling directory')
    parser.add_argument('--input_tiff', type=str, required=True, help='Path to input tiff file')
    parser.add_argument('--location_name', type=str, required=True, help='Location name of job')
    parser.add_argument('--outpath', type=str, required=True, help='Path to the output json file')
    args = parser.parse_args()
    main(args.input_file, args.model_path, args.shape_path, args.tiling_dir, args.input_tiff, args.location_name,
         args.outpath)
