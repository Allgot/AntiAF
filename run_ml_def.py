# author: EManuele/Immanuel, JLee/Allgot

import glob
import pandas
import numpy as np
import sklearn.metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib.backends.backend_pdf import PdfPages
from timeit import default_timer as timer
from multiprocessing import cpu_count
import pickle

# LLM
# import ollama

# Confusion matrix + .png + title
def plot_confusion_matrix(cm, classes, normalize=False, subplot=False, title='Confusion matrix', cmap=plt.cm.Blues, avg_scores="-", filename=None):
    if not subplot:
        plt.figure(figsize=(11.69, 8.27))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                                    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.annotate("sklearn Acuracy Score Mean:\n" + str(avg_scores) , xy=(0, 0), xycoords='axes fraction', fontsize=10,
            horizontalalignment='left', verticalalignment='bottom', xytext=(-200, 0), textcoords='offset points')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename + ".png", transparent=True)


# pie chart class instances
def plot_X_test_classes_instances(label, classes, counts, filename=None):
    fig, ax = plt.subplots(figsize=(11.69, 8.27), subplot_kw=dict(aspect="equal"))

    plt.title("#instances classes")
    # absolute = counts.sum()

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n(#{:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(counts, autopct=lambda pct: func(pct, counts),
                                      textprops=dict(color="w"))
    ax.legend(wedges, label,
              title="Classes",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    if filename is not None:
        plt.savefig(filename + ".png", transparent=True)

# returns: array of scores, matrix sum of all confusion matrix
def leave_one_subject_out_cross_validation(classifier, X_train, y_train, X_test, y_test, print_report=True, is_llm=False):
    cnf_list = []
    scores_list = []
    std_scaler = preprocessing.StandardScaler()

    print(f"X_train shape -- {X_train.shape}")
    # Use the scaler on training set
    X_train_sc = std_scaler.fit_transform(X_train)
    # and on testing set
    X_test_sc = std_scaler.transform(X_test)

    classifier.fit(X_train_sc, y_train)
    y_pred = classifier.predict(X_test_sc)
    cnf_run = confusion_matrix(y_test, y_pred)
    score_run = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("sklearn Accuracy Score:  %f " % (score_run))
    if print_report:
        print(cnf_run)
        print(classification_report(y_test, y_pred))
    scores_list.append(score_run)
    cnf_list.append(cnf_run)
    return scores_list, sum(cnf_list)


def print_classifier_results(scores_list, classes_tot_list, instances_per_class_list):
    print("sklearn Accuracy Scores: %s Avg: %2f" % (scores_list, sum(scores_list)/len(scores_list)))
    print("Classes tot: %d -> %s" % (len(classes_tot_list), str(classes_tot_list)))
    print("Counts tot: %s" % (str(instances_per_class_list)))

# We use the SAME classes passed to plot_confusion_matrix so order is the same
def stats_from_confusion_matrix_sum_macro_micro(confusion_matrix, classes, flow_timeout, activity_timemout, classifier, n_features):
    columns_stat_classes = ["APP", "FLOW_TIMEOUT", "ACTIVITY_TIMEOUT", "CLASSIFIER", "N_FEATURES",
               "TRUE_POSITIVE", "FALSE_POSITIVE", "TRUE_NEGATIVE", "FALSE_NEGATIVE",
               "PRECISION", "RECALL", "F1", "ACCURACY"]
    stat_classes = pandas.DataFrame(columns=columns_stat_classes)

    columns_per_classifier = ["CLASSIFIER", "N_FEATURES", "FLOW_TIMEOUT", "ACTIVITY_TIMEOUT", \
                            "AVERAGE_ACCURACY", "ERROR_RATE", "MICRO_PRECISION", "MICRO_RECALL", \
                            "MICRO_F1_SCORE", "MACRO_PRECISION", "MACRO_RECALL", "MACRO_F1_SCORE"]
    stat_classifier = pandas.DataFrame(columns=columns_per_classifier)
    #  a.loc[len(a)]=[1,2,3]
    True_Positive = np.diag(confusion_matrix)
    False_Positive = confusion_matrix.sum(axis=0) - True_Positive
    False_Negative = confusion_matrix.sum(axis=1) - True_Positive
    True_Negative = confusion_matrix.sum() - (True_Positive + False_Negative + False_Positive)
    Precision = True_Positive / (True_Positive + False_Positive)
    Recall = True_Positive / (True_Positive + False_Negative)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)
    Accuracy = (True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative)
    for i in range(len(classes)):
        stat_classes.loc[len(stat_classes)] = [classes[i], flow_timeout, activity_timemout, classifier, n_features,
                            True_Positive[i], False_Positive[i], True_Negative[i],
                            False_Negative[i], Precision[i], Recall[i], F1_score[i],
                            Accuracy[i]]
    # return stat_classes
    Average_Accuracy = (Accuracy.sum()) / len(classes)
    Error_Rate_per_class = (False_Positive + False_Negative ) / (True_Positive + True_Negative + False_Positive + False_Negative)
    Error_Rate = (Error_Rate_per_class.sum()) / len(classes)
    Micro_Precision = ( (True_Positive.sum()) ) / ( (True_Positive.sum()) + (False_Positive.sum()) )
    Micro_Recall = ( (True_Positive.sum()) ) / ( (True_Positive.sum()) + (False_Negative.sum()) )
    Micro_F1_score = 2 * ( Micro_Precision * Micro_Recall ) / (Micro_Precision + Micro_Recall)
    Macro_Precision = ( Precision.sum() ) / len(classes)
    Macro_Recall = ( Recall.sum() ) / len(classes)
    Macro_F1_score = 2 * ( Macro_Precision * Macro_Recall ) / (Macro_Precision + Macro_Recall)
    stat_classifier.loc[(len(stat_classifier))] = [classifier, n_features, flow_timeout, activity_timemout,
                        Average_Accuracy, Error_Rate, Micro_Precision, Micro_Recall,
                        Micro_F1_score, Macro_Precision, Macro_Recall, Macro_F1_score]
    return stat_classes, stat_classifier

# aux method, input: list of classifier stats from stats_from_confusion_matrix_sum
# output: a .csv with all stats
def stats_to_csv(list_of_datasframes_stats_from_cnf, classifier_name):
        all = pandas.concat(list_of_datasframes_stats_from_cnf).reset_index(drop=True)
        filename = classifier_name + ".csv"
        all.to_csv(filename, sep=",", index=False, encoding="utf-8")

# saves confusion matrix to pickle file
def save_cm(confusion_matrix, classes, filename):
    # save all to a list
    cm_pieces = {}
    cm_pieces['cm'] = confusion_matrix
    cm_pieces['classes'] = classes
    df_cm = pandas.DataFrame(confusion_matrix, columns=classes, index=classes)
    cm_pieces['dfcm'] = df_cm
    # save to csv df_cm
    df_cm.to_csv("cm_" + filename + ".csv")
    # save to pickle cm_pieces
    with open(filename + ".pkl", "wb") as output:
        pickle.dump(cm_pieces, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    start_main = timer()
    jobs = cpu_count() - 1
    # random seed
    seed = 5000
    # folds
    k_folds = 10
    # path to folder that contains flows .csv
    csv_path = "output/"
    csv_path_def = "output_0.60/"
    # Output filename prefix (ie. reduced_padding)
    output_filename_prefix = "def"
    # flow timeout value
    flow_timeout = 10
    # activity timeout value
    activity_timeout = 2
    # final output file name
    output_filename_prefix += "_" + str(flow_timeout) + "_" + str(activity_timeout)
    # Load data
    dataset = pandas.concat([pandas.read_csv(f) for f in glob.glob(csv_path + "/*.csv")]).reset_index(drop=True)
    dataset_def = pandas.concat([pandas.read_csv(f) for f in glob.glob(csv_path_def + "/*.csv")]).reset_index(drop=True)
    print("All .csv loaded into dataframe")
    # Dataset: 76 columns
    # Columns 0,1 -> label , category
    # Last 6 columns: IP_DST, IP_SRC, TOT_BYTES
    #                 TOT_PACKETS, TOT_OUT_PACKETS
    #                 TOT_IN_PACKETS
    # Features Columns: 76-2-6 = 68 columns
    X_all_features_train = np.array(dataset.iloc[:, 2:70])
    X_all_features_test = np.array(dataset_def.iloc[:, 2:70])
    # 0-68 All Features
    X_time_bursts_sizes_train = np.array(X_all_features_train[:, ])
    X_time_bursts_sizes_test = np.array(X_all_features_test[:, ])
    # App Label Column
    y_train = np.array(dataset.iloc[:, 0])
    y_test = np.array(dataset_def.iloc[:, 0])
    # Classes, labels, counts
    classes_tot, count_tot = np.unique(y_test, return_counts=True)

    # LLM (Few-Shot Learning) Config
    llm_model = "llama3:8b-text-q6_K"

    # Random Forest Config
    rf_all = RandomForestClassifier(n_jobs=jobs, random_state=seed, class_weight=None, n_estimators=400)

    # kNN Config
    knn_all = KNeighborsClassifier(n_jobs=jobs)

    # SVM SVC Config
    svc_all = SVC(random_state=seed, kernel='rbf', C=10, gamma=0.01)

    # XGBoost Config
    xgb_all = XGBClassifier(n_jobs=jobs, random_state=seed, n_estimators=5000,
                            learning_rate=0.01, max_depth=5, min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8, reg_alpha=1e-05, reg_lambda=0.1,
                            objective='multi:softmax', eval_metric='mlogloss')

    # LLM
    # scores_llm_all, cnf_llm_all = cross_validation_stratifiedKFold(llm_model, X_time_bursts_sizes, y, k_folds, classes_tot, is_llm=True)
    # llm_all_stats, llm_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_llm_all, classes_tot, flow_timeout, activity_timeout, "LLM", np.size(X_time_bursts_sizes, 1))

    # scores_llm_time_bursts, cnf_llm_time_bursts = cross_validation_stratifiedKFold(llm_model, X_time_bursts, y, k_folds, classes_tot, is_llm=True)
    # llm_time_bursts_stats, llm_time_bursts_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_llm_time_bursts, classes_tot, flow_timeout, activity_timeout, "LLM", np.size(X_time_bursts, 1))

    # scores_llm_time, cnf_llm_time = cross_validation_stratifiedKFold(llm_model, X_time, y, k_folds, classes_tot, is_llm=True)
    # llm_time_stats, llm_time_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_llm_time, classes_tot, flow_timeout, activity_timeout, "LLM", np.size(X_time, 1))

    # stats_to_csv([llm_all_stats, llm_time_bursts_stats, llm_time_stats], output_filename_prefix + "_" + "LLM")
    # stats_to_csv([llm_all_stats_classifier, llm_time_bursts_stats_classifier, llm_time_stats_classifier], output_filename_prefix + "_" + "LLM_micro_macro")

    # llm_timer = timer()
    # llm_time = llm_timer - start_main
    # print("LLM elapsed time: %.2f" % (llm_time))

    # Random Forest
    scores_rf_all, cnf_rf_all = leave_one_subject_out_cross_validation(rf_all, X_time_bursts_sizes_train, y_train, X_time_bursts_sizes_test, y_test)
    rf_all_stats, rf_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_rf_all, classes_tot, flow_timeout, activity_timeout,"RandomForest", np.size(X_time_bursts_sizes_test, 1))

    stats_to_csv([rf_all_stats], output_filename_prefix + "_" + "RandomForest")

    stats_to_csv([rf_all_stats_classifier], output_filename_prefix + "_" + "RandomForest_micro_macro")

    rf_timer = timer()
    # rf_time = rf_timer - llm_timer
    rf_time = rf_timer - start_main
    print("Random Forest elapsed time: %.2f" % (rf_time))

    # knn
    scores_knn_all, cnf_knn_all = leave_one_subject_out_cross_validation(knn_all, X_time_bursts_sizes_train, y_train, X_time_bursts_sizes_test, y_test)
    knn_all_stats, knn_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_knn_all, classes_tot, flow_timeout, activity_timeout, "KNN", np.size(X_time_bursts_sizes_test, 1))

    stats_to_csv([knn_all_stats], output_filename_prefix + "_" + "KNN")
    stats_to_csv([knn_all_stats_classifier], output_filename_prefix + "_" + "KNN_micro_macro")

    knn_timer = timer()
    knn_time = knn_timer - rf_timer
    print("kNN elapsed time: %.2f" % (knn_time))

    # SVM
    scores_svc_all, cnf_svc_all = leave_one_subject_out_cross_validation(svc_all, X_time_bursts_sizes_train, y_train, X_time_bursts_sizes_test, y_test)
    svc_all_stats, svc_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_svc_all, classes_tot, flow_timeout, activity_timeout, "SVC", np.size(X_time_bursts_sizes_test,1))

    stats_to_csv([svc_all_stats], output_filename_prefix + "_" + "SVC")
    stats_to_csv([svc_all_stats_classifier], output_filename_prefix + "_" + "SVC_per_micro_macro")

    svm_timer = timer()
    svm_time = svm_timer - knn_timer
    print("SVM elapsed time: %.2f" % (svm_time))

    # XGB
    scores_xgb_all, cnf_xgb_all = leave_one_subject_out_cross_validation(xgb_all, X_time_bursts_sizes_train, y_train, X_time_bursts_sizes_test, y_test)
    xgb_all_stats, xgb_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_xgb_all, classes_tot, flow_timeout, activity_timeout, "XGB", np.size(X_time_bursts_sizes_test,1))

    stats_to_csv([xgb_all_stats], output_filename_prefix + "_" + "XGB")
    stats_to_csv([xgb_all_stats_classifier], output_filename_prefix + "_" + "XGB_per_micro_macro")

    xgb_timer = timer()
    xgb_time = xgb_timer - svm_timer
    print("XGB elapsed time: %.2f" % (xgb_time))
    # print("Elapsed time - RandomForest: %.2f -- kNN: %.2f -- SVM: %.2f -- LLM: %.2f" % (rf_time, knn_time, svm_time, llm_timer))
    print("Elapsed time - RandomForest: %.2f -- kNN: %.2f -- SVM: %.2f -- XGB: %.2f" % (rf_time, knn_time, svm_time, xgb_timer))
    print("Total elapsed time: %.2f" %(xgb_timer - start_main))

    plt.ion()
    with PdfPages(output_filename_prefix + ".pdf") as pdf:
        plot_X_test_classes_instances(classes_tot, classes_tot, count_tot)
        pdf.savefig()
        plot_confusion_matrix(cnf_rf_all, classes_tot, normalize=True, title="Random Forest - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_rf_all)/len(scores_rf_all))
        pdf.savefig()
        plot_confusion_matrix(cnf_knn_all, classes_tot, normalize=True, title="kNN - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_knn_all)/len(scores_knn_all))
        pdf.savefig()
        plot_confusion_matrix(cnf_svc_all, classes_tot, normalize=True, title="svc - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_svc_all)/len(scores_svc_all))
        pdf.savefig()
        plot_confusion_matrix(cnf_xgb_all, classes_tot, normalize=True, title="XGBoost - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_xgb_all)/len(scores_xgb_all))
        pdf.savefig()
        plt.show()
