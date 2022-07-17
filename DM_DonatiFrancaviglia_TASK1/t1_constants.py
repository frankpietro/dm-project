# dataframe names
PICKLE_FOLDER = '../pickle/'
CLEAN = 'clean_'
PURE = 'pure_'
PREPARED = 'prepared_'
SELECTED = 'selected_'

BASKET_DF = 'basket_df'
USER_DF = 'user_df'
SHOP_DF = 'shop_df'
ITEM_DF = 'item_df'
DATE_DF = 'date_df'
TOTAL_DF = 'total_df'
CLUST_DF = 'clustering_df'



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

# other constants
MAX_K = 30
N_INIT = 10
MAX_ITER = 100