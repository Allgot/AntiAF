# author: EManuele/Immanuel


from sklearn.preprocessing import LabelEncoder
import glob
import pandas
import numpy as np
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib.backends.backend_pdf import PdfPages
from timeit import default_timer as timer
from multiprocessing import cpu_count
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


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
def cross_validation_stratifiedKFold(classifier, X, y, splits, labels, print_report=True):
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
    seed = 17
    # folds
    k_folds = 10
    # path to folder that contains flows .csv
    csv_path = "output_2018/"
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

    # Param_grid on all features : modulate the param_grid 
    param_grid=[{
        'n_estimators':[400,800,1200],
        #'max_features':['sqrt','log2'],
        #'min_samples_leaf':[1,2,4],
        'max_depth':[10,50,100]
        }]
    rf_all = RandomForestClassifier(n_jobs=jobs, random_state=seed)
    grid_search = GridSearchCV(rf_all,param_grid,cv=2,scoring='accuracy',n_jobs=jobs)
    grid_search.fit(X_time,y)
    print(grid_search.best_score_)
    print(grid_search.best_params_)

    print('------------------------------------------------')
    
    # Random Forest Config
    rf_time = RandomForestClassifier(n_jobs=jobs, random_state=seed, max_features='sqrt', class_weight=None, n_estimators=800)
    rf_time_bursts = RandomForestClassifier(n_jobs=jobs, random_state=seed, max_features='sqrt', class_weight=None, n_estimators=800)

    # Random Forest
    scores_rf_all, cnf_rf_all = cross_validation_stratifiedKFold(rf_all, X_time_bursts_sizes, y, k_folds, classes_tot)
    rf_all_stats, rf_all_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_rf_all, classes_tot, flow_timeout, activity_timeout,"RandomForest", np.size(X_time_bursts_sizes, 1))

    scores_rf_time_bursts, cnf_rf_time_bursts = cross_validation_stratifiedKFold(rf_time_bursts, X_time_bursts, y, k_folds, classes_tot)
    rf_time_bursts_stats, rf_time_bursts_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_rf_time_bursts, classes_tot, flow_timeout, activity_timeout,"RandomForest", np.size(X_time_bursts, 1))

    scores_rf_time, cnf_rf_time = cross_validation_stratifiedKFold(rf_time, X_time, y, k_folds, classes_tot)
    rf_time_stats, rf_time_stats_classifier = stats_from_confusion_matrix_sum_macro_micro(cnf_rf_time, classes_tot, flow_timeout, activity_timeout, "RandomForest", np.size(X_time, 1))

    stats_to_csv([rf_all_stats, rf_time_bursts_stats, rf_time_stats], output_filename_prefix + "_" + "RandomForest")
    stats_to_csv([rf_all_stats_classifier, rf_time_bursts_stats_classifier, rf_time_stats_classifier], output_filename_prefix + "_" + "RandomForest_micro_macro")

    rf_timer = timer()
    rf_time = rf_timer - start_main
    print("Random Forest elapsed time: %.2f" % (rf_time))

    plt.ion()
    with PdfPages(output_filename_prefix + ".pdf") as pdf:
        plot_X_test_classes_instances(classes_tot, classes_tot, count_tot)
        pdf.savefig()
        plot_confusion_matrix(cnf_rf_time, classes_tot, normalize=True, title="Random Forest - Confusion matrix - cross 10 StratifiedKfold Time Features", avg_scores=sum(scores_rf_time)/len(scores_rf_time))
        pdf.savefig()
        plot_confusion_matrix(cnf_rf_time_bursts, classes_tot, normalize=True, title="Random Forest - Confusion matrix -  cross 10 StratifiedKfold Time + Burst Features", avg_scores=sum(scores_rf_time_bursts)/len(scores_rf_time_bursts))
        pdf.savefig()
        plot_confusion_matrix(cnf_rf_all, classes_tot, normalize=True, title="Random Forest - Confusion matrix - cross 10 StratifiedKfold Time + Bursts + Size Features", avg_scores=sum(scores_rf_all)/len(scores_rf_all))
        pdf.savefig()
        plt.show()
