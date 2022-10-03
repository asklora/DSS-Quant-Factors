# from data_preparation.src.calculation_premium import calcPremium
from utils import models, read_query, sys_logger
from configs import LOGGER_LEVELS
from sqlalchemy import select, join, and_, not_
import pandas as pd
import openpyxl

from utils.packages.sql.src.models.factor.tables import FactorPreprocessPremium
logger = sys_logger(__name__, LOGGER_LEVELS.ANALYSIS_PREMIUM)


def cal_20_premium_average_for_all(weeks_to_expire: int = 8, 
                                   average_days: int = -7):
    """calculate the average of premium for 20 years with specified qcut percentage for each currency"""

    dflist = []

    # Create New workbook
    wb = openpyxl.Workbook()
    wb.save(filename='test.xlsx')

    for currency in ['USD', 'EUR', 'HKD', 'CNY']:
        conditions = [
            models.FactorPreprocessPremium.group.in_([currency]),
            models.FactorPreprocessPremium.weeks_to_expire.in_([weeks_to_expire]),
            models.FactorPreprocessPremium.average_days.in_([average_days])
        ]
        model = models.FactorPreprocessPremium
        query = select(model.testing_period, model.field, 
                       model.value) \
            .where(and_(*conditions))

        logger.info(f"Start downloading data for {currency}")

        df = read_query(query=query,  
                        index_cols=['field', 'testing_period'],
                        keep_index=True)

        logger.info(f"Finish downloading factor premium for {currency}")

        mean = df.groupby('field')['value'].apply(lambda x: float(round(x.mean()*10000,2))).to_frame('Mean (times 10000)')

        # mean['Average_is_positive'] = mean['Mean_with_10%_quantile'] \
        #                                 .apply(lambda m: True if (m > 0) 
        #                                  else False)
        mean['Mean_with_1%_quantile (times 10000)']=df.groupby('field')['value'].apply(lambda x: float(round(x[(x>=x.quantile(0.01))&(x<=x.quantile(0.99))].mean()*10000,2)))

        mean['Mean_with_10%_quantile (times 10000)']=df.groupby('field')['value'].apply(lambda x: float(round(x[(x>=x.quantile(0.1))&(x<=x.quantile(0.9))].mean()*10000,2)))

        mean['Mean_with_25%_quantile (times 10000)']=df.groupby('field')['value'].apply(lambda x: float(round(x[(x>=x.quantile(0.25))&(x<=x.quantile(0.75))].mean()*10000,2)))

        mean['Median (times 10000)']=df.groupby('field')['value'].apply(lambda x: float(round(x.median()*10000,2)))

        testing_period_range_grouping = df.reset_index() \
                    .groupby('field')['testing_period']

        mean['Number_of_years'] = (testing_period_range_grouping.max() \
                                     -testing_period_range_grouping.min())\
                                     .dt.days/365
                                     
        mean['Number_of_years']=mean['Number_of_years'].apply(lambda x: float(round(x,2)))
        
        mean = mean.assign(Currency=f'{currency}') 

        field = df.reset_index()['field'].drop_duplicates().to_frame()
        
        result = field.merge(mean, on='field')

        logger.info(f"Finish data processing for {currency}")

        dflist.append((currency, result))

    with pd.ExcelWriter('factor_processed_premium_2008-2019_with_20_pct_quct.xlsx', mode="a", engine="openpyxl", if_sheet_exists='replace') as writer:
        for i, result in enumerate(dflist):
            result[1].to_excel(writer, sheet_name=f"{result[0]}")
        logger.info(f"Finish saving csv")

if __name__ == "__main__":
    cal_20_premium_average_for_all()

# REM  *****  BASIC  *****

# Private Sub WorksheetLoop()
# Dim i As Integer
# Dim alphabet(1 TO 5) As String
# Dim item as variant

# alphabet(1)="C"
# alphabet(2)="D"
# alphabet(3)="E"
# alphabet(4)="F"
# alphabet(5)="G"
  
# For Each item in alphabet
# 	For i=1 to 35
# 		If Worksheets("Sheet0").Cells(item,i).Value < 0 Then
# 			Worksheets("Sheet0").Cells.Interior.ColorIndex=3
# 		End If
# 	Next i
# Next item
  
# End Sub



