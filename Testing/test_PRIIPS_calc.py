import datetime
import numpy as np
import unittest
from pandas import read_csv
from PRIIPS_calc.PRIIPS_calc import PRIIPS_stats_bootstrap, PRIIPS_stats_df, PRIIPS_stats_bootstrap, plot_bootstrap_priips

class Test_test_PRIIPS_calc(unittest.TestCase):
    def test_new_priips(self):
        ANNUALISE = 256 # Number of trading days in the year for annualising = 52*5 Actual number of days often less
# For example, PRIIPS uses 256

# Total Period
        MAX_DATE_FILTER = datetime.datetime(2023, 10, 27) # end of sample 'March 2022'
        MIN_DATE_FILTER = datetime.datetime(2004, 1, 1) # beginning of sample 'Jan 2004' - Jan 7 results in similar


        date_ranges = {
                '2004-2010' : (datetime.datetime(2004, 1, 1), datetime.datetime(2010, 12, 31)),
                '2010-2016' : (datetime.datetime(2011, 1, 1), datetime.datetime(2016, 12, 31)),
                '2017-2023' : (datetime.datetime(2017, 1, 1), datetime.datetime(2023, 10, 27))
                }
# Read in the price history
        df_price_history = read_csv('https://raw.githubusercontent.com/TimWilding/FinanceDataSet/main/PRIIPS_Data_Set.csv', parse_dates=[1])

        df_price_history['LogPrice'] = np.log(df_price_history['Close'])
        df_price_history.sort_values(['Index', 'Date'], inplace=True)
        df_price_history['LogReturn'] = 100*df_price_history.groupby('Index')['LogPrice'].diff()
        df_price_history = df_price_history[df_price_history.Date>=MIN_DATE_FILTER]
        df_sample_5y = df_price_history[(df_price_history.Date>datetime.datetime(2018, 10, 27)) & (df_price_history.Date<=datetime.datetime(2023, 10, 27))]
        df_bs = PRIIPS_stats_bootstrap(df_sample_5y, True)
        self.fail("Not implemented")

if __name__ == '__main__':
    unittest.main()
