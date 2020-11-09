import os
import numpy as np
from scipy import stats


# in
meta_path = 'compare_ppgs_inference_and_findA_and_calculateCE_list.txt'

# out
out_path = 'compare_ppgs_better_result.txt'


def dist1(ppgs_a, ppgs_b):
    ans = 0
    for i in range(ppgs_a.shape[0]):
        e = ppgs_a[i]
        c = ppgs_b[i]
        ans += np.linalg.norm(e - c)
    return ans


def dist2KL(ppgs_a, ppgs_b):
    ans = 0
    for i in range(ppgs_a.shape[0]):
        e = ppgs_a[i]
        c = ppgs_b[i]
        ans += stats.entropy(e,c) + stats.entropy(c,e)
    return ans


def dist3BetterNum(ppg_std, ppg_baseline, ppg_findA):
    # ans = []
    cnt_baseline_better_d1 = 0
    cnt_findA_better_d1 = 0
    cnt_baseline_better_d2 = 0
    cnt_findA_better_d2 = 0
    for i in range(ppg_std.shape[0]):
        d1_baseline = np.linalg.norm(ppg_std[i] - ppg_baseline[i])
        d1_findA = np.linalg.norm(ppg_std[i] - ppg_findA[i])
        if d1_baseline < d1_findA:
            cnt_baseline_better_d1 += 1
        else:
            cnt_findA_better_d1 += 1

        d2_baseline = stats.entropy(ppg_std[i],ppg_baseline[i]) + stats.entropy(ppg_baseline[i],ppg_std[i])
        d2_findA = stats.entropy(ppg_std[i],ppg_findA[i]) + stats.entropy(ppg_findA[i],ppg_std[i])
        if d2_baseline < d2_findA:
            cnt_baseline_better_d2 += 1
        else:
            cnt_findA_better_d2 += 1
    return cnt_baseline_better_d1, cnt_findA_better_d1, cnt_baseline_better_d2, cnt_findA_better_d2




def main():
    a = open(meta_path, 'r')
    f = open(out_path, 'w')
    for i, six_ppgs_speaker in enumerate(a): 
        ppg, findA_ppg, rec_ppg_en, rec_ppg_cn, findA_rec_ppg_en, findA_rec_ppg_cn, speaker = six_ppgs_speaker.strip().split('|')
        ppg = np.load(ppg)
        findA_ppg = np.load(findA_ppg)
        rec_ppg_en = np.load(rec_ppg_en)
        rec_ppg_cn = np.load(rec_ppg_cn)
        rec_ppg = np.concatenate((rec_ppg_en, rec_ppg_cn),axis=-1)

        findA_rec_ppg_en = np.load(findA_rec_ppg_en)
        findA_rec_ppg_cn = np.load(findA_rec_ppg_cn)
        findA_rec_ppg = np.concatenate((findA_rec_ppg_en, findA_rec_ppg_cn),axis=-1)

        speaker = int(speaker)

        baseline_d1 = dist1(ppg, rec_ppg)
        findA_d1 = dist1(ppg, findA_rec_ppg)
        baseline_d2 = dist2KL(ppg, rec_ppg)
        findA_d2 = dist2KL(ppg, findA_rec_ppg)

        f.write(str(i) + ' ' + 'speaker: ' + str(speaker) + '\n')
        f.write('dist1: ' + 'baseline-' + str(baseline_d1) + 'findA-' + str(findA_d1) + '\n')
        f.write('dist2KL: ' + 'baseline-' + str(baseline_d2) + 'findA-' + str(findA_d2) + '\n')

        findA_error = dist2KL(ppg, findA_ppg)
        NN_error = dist2KL(findA_ppg, findA_rec_ppg)
        f.write('findA_error: ' + str(findA_error) + 'NN_error: ' + str(NN_error) + '\n')

        cnt_baseline_d1, cnt_findA_d1, cnt_baseline_d2, cnt_findA_d2 = dist3BetterNum(ppg, rec_ppg, findA_rec_ppg) 
        f.write('d1 baseline better: ' + str(cnt_baseline_d1) + ' findA better: ' + str(cnt_findA_d1) + '\n')
        f.write('d2 baseline better: ' + str(cnt_baseline_d2) + ' findA better: ' + str(cnt_findA_d2) + '\n')

        f.write('\n')





if __name__ == '__main__':
    main()