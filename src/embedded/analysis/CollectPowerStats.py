import os
import argparse

HOURS_TO_RUN_EACH = 1.2
TIMEOUT_SECS = int(HOURS_TO_RUN_EACH * 60 * 60)
TEGRASTATS_INTERVAL_MS = 200

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path-to-listen-script', type=str, help="path to the 'Listen.py' script", required=True)
    parser.add_argument('--path-to-trained-model', type=str, help="path to the trained 2stage model", required=True)
    parser.add_argument('--path-to-powerstats-log-dir', type=str, help="directory where powerstats logs should be contained", required=True)

    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.path_to_powerstats_log_dir):
        os.mkdir(args.path_to_powerstats_log_dir)

    for half_precision in [True, False]:
        for batch_size in ["16", "64"]:
            for model_type in ["trained_model", "mobilenet", "resnet"]:

                prec_str = "float16" if half_precision else "float32"
                batch_str = f"batchsize-{batch_size}"
                model_str: str
                if model_type == "trained_model":
                    model_str = "trained-model"
                elif model_type == "resnet":
                    model_str = "resnet-101"
                else:
                    model_str = "mobilenetV2"

                logfile_name = f"{prec_str}-{model_str}-{batch_str}.txt"

                start_metrics_command = f"tegrastats --interval {TEGRASTATS_INTERVAL_MS} --logfile {args.path_to_powerstats_log_dir}/{logfile_name} &"

                listen_command = f"timeout {TIMEOUT_SECS} python3 {args.path_to_listen_script} --batch-size {batch_size}"
                if half_precision:
                    listen_command += " --half-precision"
                if model_type == "trained_model":
                    listen_command += f" --model-path {args.path_to_trained_model}"
                else:
                    listen_command += f" --random-model {model_type}"

                sleep_ten_command = "sleep 10"

                os.system(start_metrics_command)
                os.system(listen_command)
                os.system(sleep_ten_command)
                os.system("tegrastats --stop")
                print(f"Done populating {logfile_name}!")
                os.system(sleep_ten_command)


if __name__ == "__main__":
    main()