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
def cross_validation_stratifiedKFold(classifier, X, y, splits, labels, print_report=True, is_llm=False):
    cnf_list = []
    scores_list = []
    std_scaler = preprocessing.StandardScaler()

    kfold = StratifiedKFold(n_splits=splits, shuffle=True)
    print("X shape -- %s" % (X.shape,))
    for train_index, test_index in kfold.split(X, y):
        # print("TRAIN: %s TEST: %s" % (train_index, test_index))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Use the scaler on training set
        X_train_sc = std_scaler.fit_transform(X_train)
        # and on testing set
        X_test_sc= std_scaler.transform(X_test)

        if is_llm:
            # import numpy as np
            # import evaluate
            # from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

            # model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Meta-Llama-3-8B", num_labels = 10, token="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")

            # training_args = TrainingArguments(output_dir="fine_tuner", evaluation_strategy="epoch")
            # metric = evaluate.load("accuracy")

            # def compute_metrics(pred):
            #     logits, labels = pred
            #     predictions = np.argmax(logits, axis=-1)
            #     return metric.compute(predictions=predictions, references=labels)

            # trainer = Trainer(
            #     model = model,
            #     args = training_args,
            #     train_dataset = X_train_sc,
            #     eval_dataset = y_train,
            #     compute_metrics = compute_metrics
            # )

            # trainer.train()
            continue

        else:
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
    seed = 7000
    # folds
    k_folds = 10
    # path to folder that contains flows .csv
    csv_path = "output/"
    # Output filename prefix (ie. reduced_padding)
    output_filename_prefix = "test_padding"
    # flow timeout value
    flow_timeout = 10
    # activity timeout value
    activity_timeout = 2
    # final output file name
    output_filename_prefix += "_" + str(flow_timeout) + "_" + str(activity_timeout)
    # Load data
    dataset = pandas.concat([pandas.read_csv(f) for f in glob.glob(csv_path + "/*.csv")]).reset_index(drop=True)
    print("All .csv loaded into dataframe")
    # Dataset: 76 columns
    # Columns 0,1 -> label , category
    # Last 6 columns: IP_DST, IP_SRC, TOT_BYTES
    #                 TOT_PACKETS, TOT_OUT_PACKETS
    #                 TOT_IN_PACKETS
    # Features Columns: 76-2-6 = 68 columns
    X_all_features = np.array(dataset.iloc[:, 2:70])
    # 0-68 All Features
    X_time_bursts_sizes = np.array(X_all_features[:, ])
    # 0-59 Up to Burst Features
    X_time_bursts = np.array(X_all_features[:, :-9])
    # 0-23 Only time based features
    X_time = np.array(X_all_features[:, :-45])
    # App Label Column
    y = np.array(dataset.iloc[:, 0])
    # Classes, labels, counts
    classes_tot, count_tot = np.unique(y, return_counts=True)

    # LLM (Few-Shot Learning) Config
    llm_model = "llama3:8b-text-q6_K"

    # Random Forest Config
    rf_all = RandomForestClassifier(n_jobs=jobs, random_state=seed, class_weight=None, n_estimators=400)
    # rf_time_bursts = RandomForestClassifier(n_jobs=jobs, random_state=seed, class_weight=None, n_estimators=400)
    # rf_time = RandomForestClassifier(n_jobs=jobs, random_state=seed, class_weight=None, n_estimators=400)

    # kNN Config
    knn_all = KNeighborsClassifier(n_jobs=jobs)
    # knn_time_bursts = KNeighborsClassifier(n_jobs=jobs)
    # knn_time = KNeighborsClassifier(n_jobs=jobs)

    # SVM SVC Config
    svc_all = SVC(random_state=seed, kernel='rbf', C=10, gamma=0.01)
    # svc_time_bursts = SVC(random_state=seed, kernel='rbf', C=10, gamma=0.01)
    # svc_time = SVC(random_state=seed, kernel='rbf', C=10, gamma=0.01)

    # XGBoost Config
    xgb_all = XGBClassifier(n_jobs=jobs, random_state=seed, n_estimators=1000,
                            learning_rate=0.1, max_depth=5, min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8, reg_alpha=1e-05, reg_lambda=0.1,
                            objective='multi:softmax', eval_metric='mlogloss')
    
    # search_grid1 = {
    #     'max_depth':range(3,10,1),
    #     'min_child_weight':range(1,6,1)
    # }

    # grid_search1 = GridSearchCV(estimator=xgb_all, param_grid=search_grid1, n_jobs=jobs, cv=10)    
    # grid_search1.fit(X_time_bursts_sizes, y)
    # print(grid_search1.cv_results_, grid_search1.best_params_, grid_search1.best_score_)

    # search_grid2 = {
    #      'gamma': [i/10.0 for i in range(0, 5)]
    # }

    # grid_search2 = GridSearchCV(estimator=xgb_all, param_grid=search_grid2, n_jobs=jobs, cv=10, verbose=2)
    # grid_search2.fit(X_time_bursts_sizes, y)
    # print(grid_search2.cv_results_, grid_search2.best_params_, grid_search2.best_score_)

    # search_grid3 = {
    #     'subsample':[i/10.0 for i in range(6,10)],
    #     'colsample_bytree':[i/10.0 for i in range(6,10)]
    # }

    # grid_search3 = GridSearchCV(estimator=xgb_all, param_grid=search_grid3, n_jobs=jobs, cv=10, verbose=2)
    # grid_search3.fit(X_time_bursts_sizes, y)
    # print(grid_search3.cv_results_, grid_search3.best_params_, grid_search3.best_score_)

    # search_grid4 = {
    #     'reg_alpha':[0, 1e-5],
    #     'reg_lambda':[1e-1, 0.1, 1]
    # }

    # grid_search4 = GridSearchCV(estimator=xgb_all, param_grid=search_grid4, n_jobs=jobs, cv=10, verbose=2)
    # grid_search4.fit(X_time_bursts_sizes, y)
    # print(grid_search4.cv_results_, grid_search4.best_params_, grid_search4.best_score_)

    # exit(-1)

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
    scores_rf_all, cnf_rf_all = cross_validation_stratifiedKFold(rf_all, X_time_bursts_sizes, y, k_folds, classes_tot)
    rf_all_stats, rf_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_rf_all, classes_tot, flow_timeout, activity_timeout,"RandomForest", np.size(X_time_bursts_sizes, 1))

    # scores_rf_time_bursts, cnf_rf_time_bursts = cross_validation_stratifiedKFold(rf_time_bursts, X_time_bursts, y, k_folds, classes_tot)
    # rf_time_bursts_stats, rf_time_bursts_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_rf_time_bursts, classes_tot, flow_timeout, activity_timeout,"RandomForest", np.size(X_time_bursts, 1))

    # scores_rf_time, cnf_rf_time = cross_validation_stratifiedKFold(rf_time, X_time, y, k_folds, classes_tot)
    # rf_time_stats, rf_time_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_rf_time, classes_tot, flow_timeout, activity_timeout, "RandomForest", np.size(X_time, 1))

    # stats_to_csv([rf_all_stats, rf_time_bursts_stats, rf_time_stats], output_filename_prefix + "_" + "RandomForest")
    stats_to_csv([rf_all_stats], output_filename_prefix + "_" + "RandomForest")

    # stats_to_csv([rf_all_stats_classifier, rf_time_bursts_stats_classifier, rf_time_stats_classifier], output_filename_prefix + "_" + "RandomForest_micro_macro")
    stats_to_csv([rf_all_stats_classifier], output_filename_prefix + "_" + "RandomForest_micro_macro")

    rf_timer = timer()
    # rf_time = rf_timer - llm_timer
    rf_time = rf_timer - start_main
    print("Random Forest elapsed time: %.2f" % (rf_time))

    # knn
    scores_knn_all, cnf_knn_all = cross_validation_stratifiedKFold(knn_all, X_time_bursts_sizes, y, k_folds, classes_tot)
    knn_all_stats, knn_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_knn_all, classes_tot, flow_timeout, activity_timeout, "KNN", np.size(X_time_bursts_sizes, 1))

    # scores_knn_time_bursts, cnf_knn_time_bursts = cross_validation_stratifiedKFold(knn_time_bursts, X_time_bursts, y, k_folds, classes_tot)
    # knn_time_bursts_stats, knn_time_bursts_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_knn_time_bursts, classes_tot, flow_timeout, activity_timeout, "KNN", np.size(X_time_bursts, 1))

    # scores_knn_time, cnf_knn_time = cross_validation_stratifiedKFold(knn_time, X_time, y, k_folds, classes_tot)
    # knn_time_stats, knn_time_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_knn_time, classes_tot, flow_timeout, activity_timeout, "KNN", np.size(X_time, 1))

    # stats_to_csv([knn_all_stats, knn_time_bursts_stats, knn_time_stats], output_filename_prefix + "_" + "KNN")
    stats_to_csv([knn_all_stats], output_filename_prefix + "_" + "KNN")
    # stats_to_csv([knn_all_stats_classifier, knn_time_bursts_stats_classifier, knn_time_stats_classifier], output_filename_prefix + "_" + "KNN_micro_macro")
    stats_to_csv([knn_all_stats_classifier], output_filename_prefix + "_" + "KNN_micro_macro")

    knn_timer = timer()
    knn_time = knn_timer - rf_timer
    print("kNN elapsed time: %.2f" % (knn_time))

    # SVM
    scores_svc_all, cnf_svc_all = cross_validation_stratifiedKFold(svc_all, X_time_bursts_sizes, y, k_folds, classes_tot)
    svc_all_stats, svc_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_svc_all, classes_tot, flow_timeout, activity_timeout, "SVC", np.size(X_time_bursts_sizes,1))

    # scores_svc_time_bursts, cnf_svc_time_bursts = cross_validation_stratifiedKFold(svc_time_bursts, X_time_bursts, y, k_folds, classes_tot)
    # svc_time_bursts_stats, svc_time_bursts_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_svc_time_bursts, classes_tot, flow_timeout, activity_timeout, "SVC", np.size(X_time_bursts, 1))

    # scores_svc_time, cnf_svc_time = cross_validation_stratifiedKFold(svc_time, X_time, y, k_folds, classes_tot)
    # svc_time_stats, svc_time_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_svc_time, classes_tot, flow_timeout, activity_timeout, "SVC", np.size(X_time, 1))

    # stats_to_csv([svc_all_stats, svc_time_bursts_stats, svc_time_stats], output_filename_prefix + "_" + "SVC")
    stats_to_csv([svc_all_stats], output_filename_prefix + "_" + "SVC")
    # stats_to_csv([svc_all_stats_classifier, svc_time_bursts_stats_classifier, svc_time_stats_classifier], output_filename_prefix + "_" + "SVC_per_micro_macro")
    stats_to_csv([svc_all_stats_classifier], output_filename_prefix + "_" + "SVC_per_micro_macro")

    svm_timer = timer()
    svm_time = svm_timer - knn_timer
    print("SVM elapsed time: %.2f" % (svm_time))

    # XGB
    scores_xgb_all, cnf_xgb_all = cross_validation_stratifiedKFold(xgb_all, X_time_bursts_sizes, y, k_folds, classes_tot)
    xgb_all_stats, xgb_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_xgb_all, classes_tot, flow_timeout, activity_timeout, "XGB", np.size(X_time_bursts_sizes,1))

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
        # plot_confusion_matrix(cnf_llm_time, classes_tot, normalize=True, title="LLM - Confusion matrix - cross 10 StratifiedKfold Time Features", avg_scores=sum(scores_llm_time)/len(scores_llm_time))
        # pdf.savefig()
        # plot_confusion_matrix(cnf_llm_time_bursts, classes_tot, normalize=True, title="LLM - Confusion matrix - cross 10 StratifiedKfold Time + Burst Features", avg_scores=sum(scores_llm_time_bursts)/len(scores_llm_time_bursts))
        # pdf.savefig()
        # plot_confusion_matrix(cnf_llm_all, classes_tot, normalize=True, title="LLM - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_llm_all)/len(scores_llm_all))
        # pdf.savefig()
        # plot_confusion_matrix(cnf_rf_time, classes_tot, normalize=True, title="Random Forest - Confusion matrix - cross 10 StratifiedKfold Time Features", avg_scores=sum(scores_rf_time)/len(scores_rf_time))
        # pdf.savefig()
        # plot_confusion_matrix(cnf_rf_time_bursts, classes_tot, normalize=True, title="Random Forest - Confusion matrix - cross 10 StratifiedKfold Time + Burst Features", avg_scores=sum(scores_rf_time_bursts)/len(scores_rf_time_bursts))
        # pdf.savefig()
        plot_confusion_matrix(cnf_rf_all, classes_tot, normalize=True, title="Random Forest - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_rf_all)/len(scores_rf_all))
        pdf.savefig()
        # plot_confusion_matrix(cnf_knn_time, classes_tot, normalize=True, title="kNN - Confusion matrix - cross 10 StratifiedKfold Time Features", avg_scores=sum(scores_knn_time)/len(scores_knn_time))
        # pdf.savefig()
        # plot_confusion_matrix(cnf_knn_time_bursts, classes_tot, normalize=True, title="kNN - Confusion matrix -  cross 10 StratifiedKfold Time + Burst Features", avg_scores=sum(scores_knn_time_bursts)/len(scores_knn_time_bursts))
        # pdf.savefig()
        plot_confusion_matrix(cnf_knn_all, classes_tot, normalize=True, title="kNN - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_knn_all)/len(scores_knn_all))
        pdf.savefig()
        # plot_confusion_matrix(cnf_svc_time, classes_tot, normalize=True, title="svc - Confusion matrix - cross 10 StratifiedKfold Time Features", avg_scores=sum(scores_svc_time)/len(scores_svc_time))
        # pdf.savefig()
        # plot_confusion_matrix(cnf_svc_time_bursts, classes_tot, normalize=True, title="svc - Confusion matrix -  cross 10 StratifiedKfold Time + Burst Features", avg_scores=sum(scores_svc_time_bursts)/len(scores_svc_time_bursts))
        # pdf.savefig()
        plot_confusion_matrix(cnf_svc_all, classes_tot, normalize=True, title="svc - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_svc_all)/len(scores_svc_all))
        pdf.savefig()
        plot_confusion_matrix(cnf_xgb_all, classes_tot, normalize=True, title="XGBoost - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_xgb_all)/len(scores_xgb_all))
        pdf.savefig()
        plt.show()
