"""모든 GPU 단계는 큐가 선행 DINO의 성공 종료 뒤 별도 프로세스로 호출한다."""
import argparse
import multiprocessing
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "preflight", "queue", "smoke", "train", "head-nms", "head-free", "report"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    # import ultralytics 이전에 설정해 MENDI·전역 config로의 쓰기를 막는다.
    from . import common as c
    run = c.safe_run(args.run_dir)
    os.environ["YOLO_CONFIG_DIR"] = str(run / "ultralytics_config")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    if args.stage == "prepare":
        from .prepare import prepare
        prepare(run)
    elif args.stage == "queue":
        from .runner import queue
        queue(run)
    elif args.stage == "report":
        from .report import report
        report(run)
    else:
        from .gpu import execute
        execute(run, args.stage, args.batch)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
