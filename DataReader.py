# -*- coding: utf-8 -*-
# @Author: larry
# @Date:   2017-02-22 19:24:45
# @Last Modified by:   larry
# @Last Modified time: 2017-03-01 22:30:24

import pymssql
import pandas as pd
import datetime
import CommonObject
import constant


class DataReader:
    """GTA DataBase Reader"""

    def __init__(self, start, end, symbol, level, interval, periodicity):
        self.conn = pymssql.connect(
            server="10.6.88.26", user="sa", password="ts-test12345", port=1433)

        self.start = start
        self.end = end
        self.symbol = symbol
        self.level = level
        self.interval = interval
        self.pointer = start
        # dfbegin indicates where is the beginning position for data read when
        # DataReader is called
        self.dfbegin = start
        self.periodicity = periodicity

        self.SHANGHAI_STOCK_EXCHANGE = constant.Markets.SH_STOCK_EX
        self.SHENZHEN_STOCK_EXCHANGE = constant.Markets.SZ_STOCK_EX

        self.DATABASE_STOCK_LEVEL1_PRE = "GTA_SEL1_TRDMIN_"

        self.TABLE_STOCK_SHANGHAI_LEVEL1_MIN = "dbo.SHL1_TRDMIN"
        self.TABLE_STOCK_SHENZHEN_LEVEL1_MIN = "dbo.SZL1_TRDMIN"

        self.DATABASE_QIA_QDB = "GTA_QIA_QDB"

        self.TABLE_STOCK_DAILY = "dbo.STK_MKT_FWARDQUOTATION"
        self.TABLE_STOCK_WEEKLY = "dbo.STK_MKT_FWARDQUOTATIONWEEK"
        self.TABLE_STOCK_MONTHLY = "dbo.STK_MKT_FWARDQUOTATIONMONTH"

        self.TABLE_INDEX_DAILY = "dbo.IDX_MKT_QUOTATION"
        self.TABLE_INDEX_WEEKLY = "dbo.IDX_MKT_LASTQUOTATIONWEEK"
        self.TABLE_INDEX_MONTHLY = "dbo.IDX_MKT_LASTQUOTATIONMONTH"

        self.pointerOutOfRange = False

    def getDatabaseName(self):
        if self.periodicity == CommonObject.Periodicity.Secondly:
            pass
        elif self.periodicity == CommonObject.Periodicity.Minutely:
            databaseName = self.DATABASE_STOCK_LEVEL1_PRE + \
                self.pointer.strftime("%Y%m")
        elif self.periodicity == CommonObject.Periodicity.Hourly:
            pass
        elif self.periodicity == CommonObject.Periodicity.Daily:
            pass
        elif self.periodicity == CommonObject.Periodicity.Weekly:
            pass
        elif self.periodicity == CommonObject.Periodicity.Montyly:
            pass

        return databaseName

    def getTableName(self):
        if self.periodicity == CommonObject.Periodicity.Minutely:

            if self.symbol.split('-')[1] == self.SHANGHAI_STOCK_EXCHANGE:
                tableName = self.TABLE_STOCK_SHANGHAI_LEVEL1_MIN
            elif self.symbol.split('-')[1] == self.SHENZHEN_STOCK_EXCHANGE:
                tableName = self.TABLE_STOCK_SHENZHEN_LEVEL1_MIN

            tableName = tableName + \
                str(self.interval).zfill(2) + "_" + \
                self.pointer.strftime("%Y%m")

        else:
            pass

        return tableName

    def movePointerToNextPosition(self, duration):
        if duration == CommonObject.Periodicity.Minutely:
            self.pointer = self.pointer + datetime.timedelta(minutes=1)
        elif duration == CommonObject.Periodicity.Daily:
            self.pointer = self.pointer + datetime.timedelta(days=1)

    def getCompleteTableName(self):
        return self.getDatabaseName() + "." + self.getTableName()

    def getSQLConditionScript(self, duration):
        if duration == CommonObject.Periodicity.Minutely:
            return "SECCODE = '" + self.symbol.split('-')[0] + "' AND TDATE = '" + self.pointer.strftime("%Y%m%d") + "' AND MINTIME = '" + self.pointer.strftime("%H%M") + "'"
        elif duration == CommonObject.Periodicity.Daily:
            return "SECCODE = '" + self.symbol.split('-')[0] + "' AND TDATE = '" + self.pointer.strftime("%Y%m%d") + "'"

    def constructSQLScript(self, duration):
        tableName = self.getCompleteTableName()
        conditionScript = self.getSQLConditionScript(duration)
        sqlScript = "SELECT * FROM " + tableName + " WHERE " + conditionScript
        return sqlScript

    def getMinuteData(self, duration):
        while True:
            if self.pointer < self.end:
                sqlScript = self.constructSQLScript(duration)
                df = pd.read_sql(sqlScript, self.conn)
                df = df.sort_values('MINTIME')
                if len(df.index) >= 1:
                    self.movePointerToNextPosition(duration)
                    return df
                    break
                else:
                    self.movePointerToNextPosition(duration)
                    continue
            else:
                return pd.DataFrame()
                break

    def formatSequenceLength(self, duration, timeUnitDuration=1, overlap=False, begin=None):
        # dfbegin can be set manually if parameter begin is given
        if (begin is not None): self.dfbegin = begin

        # reset pointer
        self.pointer = self.dfbegin
        df = pd.DataFrame()

        # behave a data read opration firstly to find the first pointer
        # position where has data
        oldDf = self.getMinuteData(duration)
        df = df.append(oldDf, ignore_index=True)

        # after data read, pointer points to next position, it is also the
        # beginning position for next data read when datareader is called
        self.dfbegin = self.pointer
        for i in range(1, timeUnitDuration):
            oldDf = self.getMinuteData(duration)
            df = df.append(oldDf, ignore_index=True)

        # if overlap is not true, let dfbegin points to the same position as
        # pointer
        if (not overlap):
            self.dfbegin = self.pointer

        return df
