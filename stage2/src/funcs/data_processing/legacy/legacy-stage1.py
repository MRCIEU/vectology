def get_ebi_data_cleaned(
    efo_nodes: pd.DataFrame, overwrite: bool = False, verbose: bool = True
) -> pd.DataFrame:
    """
    From stage1 get_ebi_data
    """
    f = paths.paths_stage2["ebi_data_cleaned"]
    if f.exists() and not overwrite:
        logger.info(f"{f} exists")
        ebi_df = pd.read_csv(f)
    else:
        ebi_master = pd.read_csv(
            paths.paths_init["ebi_master"],
            sep="\t",
        )

        # drop some columns
        ebi_df = ebi_master[
            ["ZOOMA QUERY", "MAPPED_TERM_LABEL", "MAPPED_TERM_URI", "MAPPING_TYPE"]
        ]
        ebi_df.rename(columns={"ZOOMA QUERY": "query"}, inplace=True)
        logger.info(f"\n{utils.df_info(ebi_df)}")
        logger.info(ebi_df.shape)

        # drop rows with multiple mappings
        ebi_df = ebi_df[~ebi_df["MAPPED_TERM_URI"].str.contains(",", na=False)]
        logger.info(ebi_df.shape)
        # ebi_df = ebi_df[~ebi_df["MAPPED_TERM_URI"].str.contains(r"\|", na=False)]
        # ebi_df = ebi_df[~ebi_df["MAPPED_TERM_URI"].str.contains("\|", na=False)]
        ebi_df = ebi_df[~ebi_df["MAPPED_TERM_URI"].str.contains("|", na=False, regex=False)]
        # ebi_df = ebi_df[~ebi_df["MAPPED_TERM_URI"].str.contains("||", na=False)]
        logger.info(ebi_df.shape)

        # clean up and lowercase
        ebi_df["id"] = ebi_df["MAPPED_TERM_URI"].str.strip()
        ebi_df["query"] = ebi_df["query"].str.lower()

        # remove underscores
        ebi_df["query"] = ebi_df["query"].str.replace("_", " ")

        # drop where query and id are duplicates
        ebi_df.drop_duplicates(subset=["query", "id"], inplace=True)
        logger.info(ebi_df.shape)

        # drop nan
        ebi_df.dropna(inplace=True)
        logger.info(ebi_df.shape)

        # drop cases where query and matched text are identical
        logger.info(ebi_df.shape)
        ebi_df = ebi_df[
            ebi_df["query"].str.lower() != ebi_df["MAPPED_TERM_LABEL"].str.lower()
        ]
        logger.info(ebi_df.shape)

        # get counts of mapping type
        logger.info(f'\n{ebi_df["MAPPING_TYPE"].value_counts()}')

        # check data against efo nodes
        efo_node_ids = list(efo_nodes["efo_id"])
        ebi_ids = list(ebi_df["id"])
        missing = []
        matched = []
        for i in ebi_ids:
            match = False
            for s in efo_node_ids:
                if i in s and match is False:
                    matched.append(s)
                    match = True
            if match is False:
                missing.append(i)
        logger.info(f"Missing: {len(missing)} {missing}")

        # remove missing from ukb data
        logger.info(ebi_df.shape)
        for i in missing:
            ebi_df = ebi_df.drop(ebi_df[ebi_df["id"].str.contains(i)].index)
        ebi_df["full_id"] = matched

        # add index as ID
        ebi_df["mapping_id"] = range(1, ebi_df.shape[0] + 1)
        ebi_df.to_csv(f, index=False)
    if verbose:
        logger.info(f"\n{utils.df_info(ebi_df)}")
        logger.info(f'\n{ebi_df["MAPPING_TYPE"].value_counts()}')
    return ebi_df
