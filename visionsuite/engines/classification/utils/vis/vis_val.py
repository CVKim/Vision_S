import os.path as osp

import cv2
import numpy as np
import torch


def save_validation(
    model, dataloader, label2class, epoch, output_dir, device, preprocessor=None
):
    cnt = 0
    for batch in dataloader:
        if len(batch) == 3 or len(batch) == 4:
            for step_idx, (image, label, fname) in enumerate(
                zip(batch[0], batch[1], batch[2])
            ):
                label = str(label2class[np.argmax(np.array(label), axis=1)[0]])
                _image = np.expand_dims(image, axis=0)
                preds = model(_image)

                image = preprocessor.undo(image=image)["image"]
                # image = preprocessor.undo(image=image, key='normalize')['image']
                pred_label = str(label2class[np.argmax(np.array(preds), axis=1)[0]])

                height, width = image.shape[:2]
                if width >= 320 and width < 640:
                    font_scale = 0.3
                    thickness = 1
                elif width >= 640 and width < 1024:
                    font_scale = 0.6
                    thickness = 1
                else:
                    font_scale = 1
                    thickness = 2
                cv2.putText(
                    image,
                    "GT: " + str(label),
                    (2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 0),
                    thickness,
                )
                cv2.putText(
                    image,
                    "PRED: " + str(pred_label),
                    (int(width / 2), 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 255),
                    thickness,
                )

                if str(pred_label) == str(label):
                    cv2.putText(
                        image,
                        "OK",
                        (int(width / 2), int(height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 3,
                        (0, 204, 0),
                        thickness * 3,
                    )
                else:
                    cv2.putText(
                        image,
                        "NG",
                        (int(width / 2), int(height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 3,
                        (0, 0, 255),
                        thickness * 3,
                    )

                cv2.imwrite(
                    osp.join(
                        output_dir,
                        str(epoch) + "_" + fname + "_{}.png".format(step_idx),
                    ),
                    image,
                )
        elif len(batch) == 2:
            for step_idx, (image, label) in enumerate(zip(batch[0], batch[1])):
                # label = str(label2class[np.argmax(np.array(label), axis=1)[0]])
                label = str(label2class[label.item()])
                _image = torch.unsqueeze(image, axis=0).to(device)
                preds = model(_image)

                # image = preprocessor.undo(image, key="normalize")["image"]
                # image = preprocessor(np.transpose(image.numpy(), (1, 2, 0)))
                image = image.to('cpu')
                image = image.numpy()
                image = image.transpose((1, 2, 0))
                if preprocessor:
                    image = preprocessor(image)
                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # image = np.ascontiguousarray(image)
                pred_label = str(label2class[np.argmax(np.array(preds.cpu().detach().numpy()), axis=1)[0]])

                width = image.shape[1]
                if width < 640:
                    font_scale = 0.5
                    thickness = 1
                elif width >= 640 and width < 1024:
                    font_scale = 0.7
                    thickness = 1
                else:
                    font_scale = 1
                    thickness = 2

                cv2.putText(
                    image,
                    "GT: " + str(label),
                    (2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 0),
                    thickness,
                )
                cv2.putText(
                    image,
                    "PRED: " + str(pred_label),
                    (int(width / 2), 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 255),
                    thickness,
                )

                if str(pred_label) == str(label):
                    cv2.putText(
                        image,
                        "OK",
                        (width - 80, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale*2,
                        (0, 204, 0),
                        thickness*2,
                    )
                else:
                    cv2.putText(
                        image,
                        "NG",
                        (width - 80, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale*2,
                        (0, 0, 255),
                        thickness*2,
                    )

                cv2.imwrite(
                    osp.join(output_dir, str(epoch) + "_{}.png".format(cnt)), image
                )
                cnt += 1