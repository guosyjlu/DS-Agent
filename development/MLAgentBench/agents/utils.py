#### Utils: Delete unnecessary infos in the log.

def clean_log(log):
    filter_patterns = [
        "[LightGBM] [Info]",
        "You can set `force_col_wise=true` to remove the overhead."
    ]
    logs = log.split('\n')
    new_log = ""
    for log in logs:
        flag = False
        for filter in filter_patterns:
            if filter in log:
                flag = True
        if flag: continue
        new_log += log
        new_log += '\n'
    return new_log
        

if __name__ == '__main__':
    logs = """
[LightGBM] [Info] Total Bins 164536
[LightGBM] [Info] Number of data points in the train set: 34369, number of used features: 672
[LightGBM] [Info] Start training from score -0.001379
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.199110 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 164536
[LightGBM] [Info] Number of data points in the train set: 34369, number of used features: 672
[LightGBM] [Info] Start training from score -0.001670
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.189127 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 164536
[LightGBM] [Info] Number of data points in the train set: 34369, number of used features: 672
[LightGBM] [Info] Start training from score -0.000494
Final MSE on validation set: 0.38225823201676384, Final MAE on validation set: 0.3323644412574187.
"""
    print(clean_log(logs))