"""
Revise various key results of stage1 into "stage1-cache" dir
"""
import pandas as pd
from metaflow import FlowSpec, Parameter, step

from funcs import info_revised, paths
from funcs.data_processing import mapping_routine, stage1_processing

from icecream import ic  # noqa

REVISED_CACHE = paths.stage2["stage1-cache"]


class ReviseStage1(FlowSpec):

    OVERWRITE = Parameter(
        "overwrite",
        help="overwrite",
        default=False,
        is_flag=True,
    )

    @step
    def start(self):
        self.model_collection = info_revised.model_collection
        self.efo_nx = stage1_processing.get_efo_nx()
        self.ebi_df = stage1_processing.get_ebi_data()
        self.next(self.make_top100)

    @step
    def make_top100(self):
        for k, v in self.model_collection.items():
            ic(k)
            cache_path = v["pairwise_filter"]
            assert cache_path.exists(), cache_path
            top_100_path = v["top_100"]
            if not top_100_path.exists() or self.OVERWRITE:
                print("Make top_100_df")
                top_100_df = stage1_processing.get_top100_using_pairwise_file(
                    cache_path=cache_path,
                    efo_nx=self.efo_nx,
                    ebi_df=self.ebi_df,
                )
                top_100_df.to_csv(top_100_path, index=False)
            else:
                print("skip")
        self.next(self.make_weighted_average)

    @step
    def make_weighted_average(self):
        top1_file_path = REVISED_CACHE / "weighted_average_top1_df.csv"
        top10_file_path = REVISED_CACHE / "weighted_average_top10_df.csv"

        print("top 1 results")
        if not top1_file_path.exists() or self.OVERWRITE:
            print("make results")
            weighted_average_top1_df = mapping_routine.prep_weighted_average_df(
                model_collection=self.model_collection, top_num=1, ebi_df=self.ebi_df
            )
            weighted_average_top1_df.to_csv(top1_file_path, index=False)
        else:
            print("skip")

        print("top 10 results")
        if not top10_file_path.exists() or self.OVERWRITE:
            print("make results")
            weighted_average_top10_df = mapping_routine.prep_weighted_average_df(
                model_collection=self.model_collection, top_num=10, ebi_df=self.ebi_df
            )
            weighted_average_top10_df.to_csv(top10_file_path, index=False)
        else:
            print("skip")

        self.next(self.make_mapping_agg)

    @step
    def make_mapping_agg(self):
        mapping_agg_df_path = REVISED_CACHE / "mapping_agg_batet_1.0.csv"
        print("mapping_agg")
        if not mapping_agg_df_path.exists() or self.OVERWRITE:
            cache_dir = REVISED_CACHE / "mapping_agg_intermediates"
            cache_dir.mkdir(exist_ok=True)
            mapping_agg_df = mapping_routine.prep_trait_efo_mapping_agg(
                ebi_data=self.ebi_df,
                model_collection=self.model_collection,
                batet_score=1.0,
                cache_dir=cache_dir,
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
    ReviseStage1()
