def start_on_update(check_interval=60, table_names=None, report_only=True):
    """ check if data tables finished ingestion -> then start

    Parameters
    ----------
    check_interval (Int): wait x seconds to check again
    table_names (List[Str]): List of table names to check whether update finished within past 3 days)
    report_only (Bool): if True, will report last update date only (i.e. will start training even no update done within 3 days)
    """
    waiting = True
    while waiting:
        update_time = read_table("ingestion_update_time", db_url_alibaba_prod)
        update_time = update_time.loc[update_time['tbl_name'].isin(table_names)]
        if report_only:
            to_slack("clair").df_to_slack("=== Ingestion Table last update === ", update_time.set_index("tbl_name")[['finish', 'last_update']])
            break

        if all(update_time['finish'] == True) & all(
                update_time['last_update'] > (dt.datetime.today() - relativedelta(days=3))):
            waiting = False
        else:
            logger.debug(f'Keep waiting...Check again in {check_interval}s ({dt.datetime.now()})')
            time.sleep(check_interval)

    to_slack("clair").message_to_slack(f" === Start Factor Model for weeks_to_expire=[{args.weeks_to_expire}] === ")
    return True


def write_db(stock_df_all, score_df_all, feature_df_all):
    """ write score/prediction/feature to DB """

    # update results
    try:
        upsert_data_to_database(stock_df_all, result_pred_table, how="append", verbose=-1, dtype=pred_dtypes)
        upsert_data_to_database(score_df_all, result_score_table, how="update", verbose=-1, dtype=score_dtypes)
        upsert_data_to_database(feature_df_all, feature_importance_table, how="append", verbose=-1, dtype=feature_dtypes)
        return True
    except Exception as e:
        # save to pickle file in local for recovery
        stock_df_all.to_pickle('cache_stock_df_all.pkl')
        score_df_all.to_pickle('cache_score_df_all.pkl')
        feature_df_all.to_pickle('cache_feature_df_all.pkl')

        to_slack("clair").message_to_slack(f"*[Factor] ERROR [FINAL] write to DB*: {e.args}")
        return False