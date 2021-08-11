from enum import Enum


class DataSource(str, Enum):
    gwas = "gwas"
    ukbb = "ukbb"
