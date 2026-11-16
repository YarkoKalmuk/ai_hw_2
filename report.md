---
title: Homework 2 - Multiple Object Video Detection
author: Yulian Zaiats and Kalmuk Yaropolk
course: Artificial Intelligence · UCU
date: 2026-11-16
---
# Homework 2: Multiple Object Video Detection

**Team:** Yulian Zaiats and Kalmuk Yaropolk

## Dataset and preprocessing

Our task is to locate and classify eight types of people and vehicles in VisDrone drone videos. Cars and people dominate the annotations, while buses are rare, so overall results can hide differences between classes. Many objects also occupy only a few pixels after resizing, particularly in crowded scenes. Our [dataset analysis and ground-truth figures](homework2.ipynb) therefore motivated smaller anchors, higher-resolution inference, and crop-based training.

| Dataset observation | Recorded value |
|---|---|
| Frames: training / validation / test | 24,201 / 2,846 / 3,811 |
| Training class shares | person 29.7%; bicycle 3.6%; car 45.7%; van 4.2%; truck 2.8%; tricycle 3.7%; bus 0.9%; motor 9.3% |
| Mean objects/frame: training / validation | 45.7 / 40.1; maxima 180 / 116 |
| Validation sequence means | 13.9–103.4 objects/frame |
| Resolutions | Training 1344×756–2720×1530; validation up to 3840×2160; test widths up to 1920 |
| Training boxes below area 32² | 24.9% originally; estimated 50.9% after resizing to long side 1333 |
| Validation buses | 264 boxes from one clip |

Before training, we checked image sizes, box bounds, class mappings, and agreement between YOLO and COCO annotations, including their coordinate conversions. Pedestrian/people and tricycle variants were merged into the target classes, while ignored/others categories became ignore regions after duplicate COCO entries were removed.

We masked ignored regions during training so that unlabelled objects would not be learned as background, removing boxes with at least half their area inside a region. Validation images remained unmasked, while the evaluator used the supplied ignore annotations through COCO crowd handling. Training masking prevents learning unlabelled objects as background; evaluation handling prevents detections in ignored regions from being counted as ordinary errors. RGB values were scaled to [0,1], followed by pretrained normalization for the convolutional models, while RT-DETR skipped mean/std normalization. Baseline augmentation used horizontal flips with probability 0.5.

## Evaluation and validation

Average Precision (AP) summarizes how well detections balance correct predictions and missed objects across confidence thresholds. A predicted box counts as a match when its overlap with the target is high enough, measured by Intersection over Union (IoU). We report COCO mean Average Precision (mAP), averaging across classes and IoU thresholds from 0.50 to 0.95. AP50 and AP75 use individual thresholds, while AP_small covers small objects; higher overlap requirements demand tighter boxes. Speed is measured in frames per second (FPS), excluding disk reads.

We compared models on the official validation set, using smaller screening sets to reject weak candidates before full evaluation. Public leaderboard scores use different data and are reported separately.

Since neighbouring video frames show almost the same scene, randomly splitting them could make validation overly optimistic. We therefore grouped clips by source video for five-fold cross-validation, keeping official validation outside the folds. This additional check covered only the selected FCOS configuration, rather than every detector.

Each held-out video was predicted by a model that had not trained on it, producing out-of-fold (OOF) predictions. Their pooled score differs from the fold average because AP is recomputed from combined rankings, rather than averaged across separate evaluations.

| Evaluation detail | Setting or result |
|---|---|
| Detection limit | 500 per image |
| Epoch curves / fusion screening | Every tenth validation frame / every tenth sorted image per clip |
| Proposal screening | 40 evenly spaced frames per clip; 280 total |
| Additional validation checks | 1,286 frames with width ≤1920; clip exclusions |
| Fold construction | 48 training sources; best balance across 2,000 seeded assignments; official validation excluded |
| OOF coverage | Every fifth held-out frame; 4,822 disjoint frames |
| Fold mean ± standard deviation / pooled OOF mAP | 0.1927 ± 0.0311 / 0.1863 |
| Official-validation mAP: fold-model mean / all-video FCOS | 0.1897 / 0.1990 |

## Detectors and training choices

We first trained four detector families to compare different ways of locating objects under a shared, limited training budget. Faster R-CNN, RetinaNet, and FCOS use convolutional neural networks (CNNs) with ResNet-50 and a Feature Pyramid Network (FPN), which supplies features at several scales. RT-DETR also has a convolutional backbone, but forms its predictions through object queries instead.

| Shared comparison settings | Value |
|---|---|
| Initialization and budget | COCO pretraining; 5 epochs; batch 2; long side 1333 |
| Sampling and augmentation | Rotating stride-5 frames; horizontal flips |
| Training | Mixed precision; seed 2026; 500-step warm-up; tenfold rate drops near epochs 4 and 5 |
| CNN optimizer | Stochastic gradient descent (SGD); momentum 0.9; weight decay 0.0001 |
| Learning rates | Faster R-CNN 0.0025; RetinaNet/FCOS 0.00125 |
| RT-DETR optimizer | AdamW; rate 0.0001; backbone 0.00001; weight decay 0.0001; gradient clipping 0.1 |

**Faster R-CNN** first identifies candidate object regions through its Region Proposal Network (RPN), then classifies and adjusts them in the region-of-interest stage. Since proposals start from predefined reference boxes called anchors, we tried smaller anchors to better match the dataset's small objects. This improved small-object detection and overall performance despite a slight decline under stricter overlap requirements, so we retained the change. A higher learning rate was tested separately rather than combined with the anchor change.

**RetinaNet** predicts directly from image features, without a second refinement stage, so its focal loss prevents easy background examples from dominating training. We increased the weight of positive examples to test whether this helped crowded scenes and retained the small gain. However, smaller anchors did not transfer successfully from Faster R-CNN: more boxes met the loose overlap requirement, but fewer met the stricter one. The missing refinement stage may help explain this difference, although the experiment does not prove that cause.

**FCOS** avoids predefined anchor shapes by assigning objects to feature-map locations inside their boxes, centre regions, and size ranges. Crowded objects can compete for these locations, so we reduced the centre sampling radius to make assignments less ambiguous. This improved small-object detection and gave us the strongest single-model result in the initial detector comparisons (sections 2–5).

We next replaced Generalized IoU (GIoU) loss with Distance IoU (DIoU), which penalizes displaced box centres more directly. Its gain was marginal and stricter box accuracy stayed unchanged, so we do not consider it a major improvement. The saved synthetic check also verifies finite DIoU loss and gradients for a zero-size predicted box.

**RT-DETR** uses object queries, candidate representations that its decoder turns into detections. During training, bipartite matching assigns each target to one prediction, while unmatched predictions are trained as background. This discourages duplicate detections, so DETR-style models generally omit conventional non-maximum suppression (NMS), which removes overlapping duplicates. Matching does not guarantee duplicate-free predictions.

We increased the query count to give crowded scenes more prediction capacity, although the gain was small under our short training schedule. Reducing resolution also removed detail needed by small objects. Section 5's epoch- and time-based curves show an early validation peak, followed by weaker results than FCOS; they do not establish fully converged performance. The zero-recall diagnostic at confidence 0.5 reflects low confidence scores rather than zero mAP.

### Detector results

The table highlights the main full-validation comparisons; dashes indicate metrics omitted here or unrecorded public scores. Notebook sections 2–6.1 also record the separate learning-rate trial, lower-resolution CNN runs, earlier longer Faster R-CNN run, and default-radius DIoU ablation. Lower resolution and longer baseline training did not improve results; default-radius DIoU slightly hurt. Anchors 8–128 received coverage analysis only. Section 6.3 corrects the baseline's slowdown-affected speed log; both baseline and smaller-anchor Faster R-CNN run at about 15 FPS.

| Configuration | mAP | AP50 | AP75 | AP_small | FPS | Public |
|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN baseline | 0.1941 | 0.3970 | 0.1664 | 0.0924 | ~15 | — |
| Faster R-CNN anchors 16–256 | 0.1969 | 0.4163 | 0.1638 | 0.1016 | ~15 | — |
| RetinaNet baseline | 0.1801 | 0.3471 | 0.1663 | — | — | — |
| RetinaNet alpha 0.5 | 0.1811 | — | — | 0.0645 | 15.7 | — |
| RetinaNet anchors 16–256 | 0.1628 | 0.3600 | 0.1253 | — | — | — |
| FCOS baseline | 0.1888 | — | — | 0.0865 | — | — |
| FCOS radius 1.0 | 0.1985 | — | 0.1742 | 0.0925 | — | — |
| RT-DETR baseline, 300 queries | 0.1429 | — | — | 0.0747 | — | — |
| RT-DETR 500 queries | 0.1488 | — | — | 0.0756 | — | — |
| RT-DETR resolution 640 | 0.0807 | — | — | 0.0226 | — | — |
| FCOS DIoU, radius 1.0 (+0.0005) | 0.1990 | 0.4012 | 0.1742 | 0.0930 | 15.0 | 0.2046 |

## Additional experiments

After tuning individual detectors, we tested whether changing their training samples or inference process could improve small-object detection. Higher-resolution FCOS inference preserved more detail but missed our predefined improvement target, while additional training reduced performance. Section 6.5 nevertheless exported the higher-resolution candidate because it had the best validation score among those options.

We then mixed whole frames with crops centred mainly on small objects, preserving local detail without completely losing scene context. The first crop pass helped slightly, but a separate follow-up pass reduced performance and the third was skipped. Tiled inference was also rejected because its increase in loose matches came with worse precise box accuracy.

Proposal diagnostics offered a more targeted change: retaining more Faster R-CNN proposals improved recall, although processing the extra candidates slowed inference. We accepted that trade-off; the speed comparison below uses the same crop-pass-1 checkpoint with different proposal budgets.

Since individual improvements remained limited, we combined detectors whose predictions could complement each other. The gains were larger than those from most single-model changes, but looser pairing, duplicate removal, and extra Faster R-CNN weight did not help. Longer class-aware training also failed, although its simultaneous setting changes prevent us from attributing the outcome to one factor.

Crop pass 1 used equal shares of whole frames and 640-pixel crops, with learning rate 0.00025. The following scores use only the fusion screening subset; outcomes apply within each comparison stage.

| Experiment | Screening mAP | Outcome |
|---|---:|---|
| Pair control: equal Faster R-CNN/FCOS weights, pairing IoU 0.55 | 0.2228 | Reference |
| Pair with looser pairing, IoU 0.45 | 0.2228 | Rejected: no gain |
| Pair with duplicate removal, NMS IoU 0.75 | 0.2228 | Rejected: no gain |
| Pair favouring Faster R-CNN, weight 0.65 | 0.2207 | Rejected: lower score |
| Add RetinaNet to the pair, existing-pair weight 0.8 | 0.2296 | Selected third member |
| Add RT-DETR to the pair, existing-pair weight 0.8 | 0.2265 | Rejected in favour of RetinaNet |
| Add RT-DETR to the trio, existing-trio weight 0.85 | 0.2328 | Selected fourth member |
| Reweight pair/RetinaNet to 0.65/0.35 instead | 0.2288 | Rejected: lower score |
| Add lighter RT-DETR, existing-trio weight 0.93 | 0.2310 | Rejected: lower score |

Full-validation milestones from sections 6.5–6.11 follow. Dashes mean other metrics are omitted here or no public score was recorded; they do not mean zero. The rejected fine-tuning epochs, second crop pass, and second-epoch RT-DETR replacement remain recorded in those sections, alongside the screening details above.

| Configuration | mAP | Other metrics | Public |
|---|---:|---|---:|
| FCOS inference resolution 1600 | 0.2002 | — | 0.1876 |
| Faster R-CNN crop pass 1 | 0.1983 | AP50 0.4181; AP75 0.1641; 15.5* FPS | 0.2258 |
| Crop pass 1, tiled inference | 0.1853 | AP50 0.4317; AP75 0.1319 | — |
| Crop pass 1, RPN budget 3000 | 0.1993 | 11.7* FPS | — |
| Faster R-CNN + FCOS | 0.2240 | — | — |
| Add RetinaNet | 0.2291 | — | — |
| Add section 5 RT-DETR | 0.2310 | — | — |
| Replace RT-DETR with retrain epoch 1 | 0.2322 | — | 0.2570 |
| Class-aware Faster R-CNN replacement, epoch 1 | 0.2272 | — | — |
| Class-aware Faster R-CNN replacement, epoch 2 | 0.2283 | — | — |
| Class-aware Faster R-CNN replacement, epoch 3 | 0.2283 | — | — |
| Four-detector chain, low threshold and flips | 0.2350 | — | — |
| Saved chain plus three RT-DETR checkpoints | 0.2394 | — | — |
| Selected two-group ensemble | **0.2516** | — | **0.3073** |

*FPS entries marked with an asterisk use the same proposal-budget comparison.*

The [Faster R-CNN failure figure](homework2.ipynb) in section 2 helps put these results in context. It shows 55 missed objects in the night scene `uav0000117_02622_v/304`, 21 missed small objects in `uav0000182_00000_v/190`, and nine wrong-class detections in `uav0000339_00001_v/58`. Low light, lost detail after resizing, and similar-looking classes are plausible explanations, but these examples do not prove the causes.

## Final ensemble

Different prediction mechanisms may produce complementary detections, which motivated combining architectures; this is not a proven explanation of every gain. Our final solution first combines Faster R-CNN, FCOS, and RetinaNet predictions, separately combines four RT-DETR checkpoints, and then merges the two groups. This arrangement lets us adjust their different confidence-score scales and performed better on validation than simply extending a single chain, although the comparison does not isolate the cause.

Each detector also sees a horizontally mirrored image as test-time augmentation (TTA), with boxes mapped back before fusion. Matching predictions reinforce one another, while unmatched boxes remain eligible so that a detection need not be found by every model.

### Exact fusion settings

In `fuse(A,B,w)`, the weights scale confidence scores before same-class boxes are paired by reciprocal best overlap at IoU ≥0.55. Matched coordinates are averaged using those scaled scores, which are then summed; unmatched boxes keep their coordinates and scaled confidence. Each call retains the top 500 without extra NMS, making the fusion order significant.

| Step | Configuration |
|---|---|
| Views | Original + horizontal flip, equal weights; confidence threshold 0.001 |
| FCOS exception | Reuse original predictions at 0.05: these already retained 500 boxes/image |
| CNN order | Crop-pass-1 Faster R-CNN, RPN pre/post-NMS budgets 3000; radius-1.0 DIoU-trained FCOS; RetinaNet alpha 0.5 |
| CNN weights | Existing group receives 0.5 when adding FCOS, then 0.8 when adding RetinaNet |
| RT-DETR order | Retrain epochs 1 and 2; section 5's 300-query checkpoint; section 5's 500-query checkpoint |
| RT-DETR weights | Existing-group weights 1/2, 2/3, 3/4 nominally balance checkpoints, subject to sequential matching/truncation |
| Final merge | CNN 0.25; RT-DETR 0.75; score weights also influence matched coordinates, not box counts |

Final mAP is **0.2516** on full validation and **0.2752** on widths ≤1920. The public leaderboard score is **0.3073**, compared with the recorded public baseline of 0.2216.

## Conclusion

Comparing the four detector families showed that changes suited to one model did not always transfer to another. Small objects motivated most experiments because resizing often removed useful detail, but smaller anchors, crops, and extra proposals produced only limited individual gains. Lower resolution, tiled inference, and longer training also showed that more speed, coverage, or computation did not necessarily improve box accuracy. The two-group ensemble gave our best validation result, so we selected it over the individual models and simpler fusion chains. However, repeated validation tuning and the short training schedule limit how broadly we can generalize these findings.

## Reproduction and limitations

**Environment.** Use Python 3.12.8 and an NVIDIA GPU supporting CUDA 11.8 wheels:

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.57.1 numpy==2.2.4 pandas==2.2.3 matplotlib==3.10.1 pycocotools==2.0.11 PyYAML==6.0.3 Pillow==11.2.1 ipython jupyterlab
jupyter lab homework2.ipynb
```

**Dataset.** Run from the notebook directory. Put `images/`, `labels/`, supplied `ignore/` folders, `annotations_coco.json`, and `data.yaml` under both `data/train/` and `data/val/`. Also provide `data/test/images/` and `data/test_images.csv`. Optional `data/sample_submission.csv` was missing in the saved check. TorchVision COCO weights and `PekingU/rtdetr_r50vd` download initially.

**Complete pipeline.** Run cells in order, allowing numbered `outputs/` folders to cache results and checkpoints. The notebook records about 25 hours on an RTX 3070 Ti Laptop, using seed 2026 and seed 9026 for the unsuccessful overnight experiment. GPU results may vary.

**Final predictions.** There is no standalone inference command: section 6.10 requires the seven checkpoints listed in `VARIANT_MODELS`, earlier definitions, and saved original FCOS predictions. Section 6.11 then fuses the saved predictions into `outputs/submissions/submission_tta_rtdetr_group.csv`, checking test-image coverage and the detection limit.

The notebook describes an unsuccessful Complete IoU (CIoU) check, but its original test and output are unavailable, so we cannot verify it as an experiment. Repeated tuning on official validation remains a limitation. This checkout lacks data, checkpoints, generated artifacts, template README, Pages configuration, and an in-repository rubric. Fresh reproduction, template compliance, Pages publication, public access, collaborator roles, and meaningful commits remain unverified.

Sources: [TorchVision reference/code](https://github.com/pytorch/vision/tree/v0.22.1/references/detection), [RT-DETR checkpoint](https://huggingface.co/PekingU/rtdetr_r50vd), [DIoU paper](https://arxiv.org/abs/1911.08287), [COCO API](https://github.com/cocodataset/cocoapi).
