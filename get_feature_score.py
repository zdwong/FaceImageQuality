
import cv2
import argparse
import pprint
import numpy as np
import tqdm
from face_image_quality import InsightFace, SERFIQ, get_embedding_quality


def get_feature_score(img_file, feat_file, score_file):
    # Create the InsightFace model
    insightface_model = InsightFace()
    # Create the SER-FIQ Model
    ser_fiq = SERFIQ()
    feat_fl = open(feat_file, 'w')
    score_fl = open(score_file, 'w')
    with open(img_file, 'r') as img_fl:
        lines = img_fl.readlines()
        for line in tqdm.tqdm(lines):
            path_image = line.strip()
            img = cv2.imread(path_image)
            # BGR->RGB and transpose (112, 112, 3) -> (3, 112, 112)
            nimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            timg = np.transpose(nimg, (2, 0, 1))
            embedding, score = get_embedding_quality(timg,
                                                     insightface_model,
                                                     ser_fiq,
                                                     use_preprocessing=False)
            score_fl.write('{} {:.4f}\n'.format(path_image, score))

            # save face embedding feature
            feat_fl.write('{} '.format(path_image))
            for e in embedding:
                feat_fl.write('{} '.format(e))
            feat_fl.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_file',
        default='/face/wong/data/ms1m_eval/lfw/img.list',
        type=str,
        help='root path'
    )
    parser.add_argument(
        '--feat_file',
        default='/face/wong/eval_feats/ser_fiq/lfw_feat.list',
        type=str,
        help='feat list'
    )
    parser.add_argument(
        '--score_file',
        default='/face/wong/eval_feats/ser_fiq/lfw_score.list',
        type=str,
        help='score list'
    )
    args = parser.parse_args()
    pprint.pprint(vars(args))

    return args


if __name__ == "__main__":
    args = parse_args()
    get_feature_score(args.img_file, args.feat_file, args.score_file)
