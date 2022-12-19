"""
Equivalent of stage1 key results in stage2
"""
import pandas as pd
from metaflow import FlowSpec, Parameter, step

from funcs import info_stage2, paths
from funcs.data_processing import mapping_routine, stage1_processing

from icecream import ic  # noqa

STAGE2_CACHE = paths.stage2["output"]


class Stage2Results(FlowSpec):

    OVERWRITE = Parameter(
        "overwrite",
        help="overwrite",
        default=False,
        is_flag=True,
    )

    VERBOSE = Parameter(
        "verbose",
        help="verbose",
        default=False,
        is_flag=True,
    )

    @step
    def start(self):
        self.model_collection = info_stage2.model_collection
        self.efo_nx = stage1_processing.get_efo_nx()
        self.ebi_df = stage1_processing.get_ebi_data()
        self.next(self.make_weighted_average)

    @step
    def make_weighted_average(self):

        top1_file_path = STAGE2_CACHE / "weighted_average_top1_df.csv"
        top10_file_path = STAGE2_CACHE / "weighted_average_top10_df.csv"

        print("top 1 results")
        if not top1_file_path.exists() or self.OVERWRITE:
            print("make results")
            weighted_average_top1_df = mapping_routine.prep_weighted_average_df_new(
                model_collection=self.model_collection, top_num=1
            )
            weighted_average_top1_df.to_csv(top1_file_path, index=False)
        else:
            print("skip")

        print("top 10 results")
        if not top10_file_path.exists() or self.OVERWRITE:
            print("make results")
            weighted_average_top10_df = mapping_routine.prep_weighted_average_df_new(
                model_collection=self.model_collection, top_num=10
            )
            weighted_average_top10_df.to_csv(top10_file_path, index=False)
        else:
            print("skip")

        self.next(self.make_mapping_agg)

    @step
    def make_mapping_agg(self):
        mapping_agg_df_path = STAGE2_CACHE / "mapping_agg_batet_1.0.csv"
        print("mapping_agg")
        if not mapping_agg_df_path.exists() or self.OVERWRITE:
            cache_dir = STAGE2_CACHE / "mapping_agg_intermediates"
            cache_dir.mkdir(exist_ok=True)
            mapping_agg_df = mapping_routine.prep_trait_efo_mapping_agg(
                ebi_data=self.ebi_df,
                model_collection=self.model_collection,
                batet_score=1.0,
                cache_dir=cache_dir,
                verbose=self.VERBOSE,
            )
            mapping_agg_df.to_csv(mapping_agg_df_path, index=False)
        else:
            mapping_agg_df = pd.read_csv(mapping_agg_df_path)
        print(mapping_agg_df)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    Stage2Results()
