from ultralytics import YOLO

def main():

    model = YOLO("../experiments/stage4_overall/weights/best.pt")

    model.train(
        data="../datasets/neu.yaml",
        epochs=25,
        patience=10,
        #imgsz=640,
        imgsz=1024,
        batch=4,

        lr0=1e-4,
        lrf=1e-3,
        mosaic=0.0,

        #freeze=10,


        amp=False,
        project="../experiments",
        name="stage3_refine_s4"
    )


if __name__ == "__main__":
    main()
