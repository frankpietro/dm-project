# dataframe names
from ast import LShift
from ctypes.wintypes import MSG
from sklearn.semi_supervised import LabelSpreading


PICKLE_FOLDER = '../pickle/'
CLEAN = 'clean_'
PURE = 'pure_'
SELECTED = 'selected_'
LABELLED = 'labelled_'

USER_DF = 'user_df'
CLUST_DF = 'clustering_df'

TRAIN = 'train_'
TEST = 'test_'

X = 'x'
Y = 'y'

# COLUMNS
# existing
ITEM = 'item_id'
CAT = 'item_category_id'
DATE = 'date'
SHOP = 'shop_id'
PUNIT = 'item_price'
CNT = 'item_cnt_day'
USER = 'user_id'
BASKET = 'basket_id'
INAME = 'item_name'
CNAME = 'item_category_name'

# new
TMP = 'temp_id'
PSUM = 'total_price'
IDCNT = 'item_dist_count'
ICNT = 'item_count'
BCNT = 'basket_count'
PMAX = 'price_max'
PAVG = 'price_avg'
PMIN = 'price_min'
PVAR = 'price_var'
CCNT = 'category_count'
MIXB = 'max_items_per_b'
MIDXB = 'max_item_dist_per_b'
SCNT = 'shop_count'
AIXB = 'avg_items_per_basket'
ABXVD = 'avg_baskets_per_d'
PE = 'price_entropy'
SE = 'shop_entropy'
IE = 'item_entropy'
CE = 'category_entropy'
NAPR = 'natural_average_price_ranking'
NABR = 'natural_average_basket_ranking'
NAIR = 'natural_average_item_ranking'
FAPR = 'frequency_average_price_ranking'
FABR = 'frequency_average_basket_ranking'
FAIR = 'frequency_average_item_ranking'
LAB = 'label'
WLAB = 'all'
SLAB = 'spending_label'

# spending labels
LS = 'L'
MS = 'M'
HS = 'H'
# LS = -1
# MS = 0
# HS = 1
# other constants
MAX_K = 30
N_INIT = 10
MAX_ITER = 100

CRT = 'criterion'
MVS = 'mean_val_score'
SVS = 'std_val_score'
MID = 'min_impurity_decrease'
MSL = 'min_samples_leaf'
MSS = 'min_samples_split'
NEST = 'n_estimators'
NNBR = 'n_neighbors'
WGH = 'weights'
KRL = 'kernel'
CPRM = 'c_param'
GMM = 'gamma'
KFF = 'coefficient'
DEG = 'degree'