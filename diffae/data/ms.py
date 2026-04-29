from pathlib import Path

from diffae.data.wmh import WMHDataset


class MSPublicDataset(WMHDataset):
    """Behaves the same as WMHDataset, but with different subsets."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._subset_names = None
        super().__init__(*args, **kwargs)

    @property
    def subset_names(self):
        """Use all subsets"""
        if self._subset_names is None:
            # get all patient directories
            all_patient_dirs = [_p for _p in self.data_dir.iterdir() if _p.is_dir()]
            # remove the last part of the name (e.g. "Ams-GE3T-1" -> "Ams-GE3T")
            # indicating the patient ID to obtain the subset name
            self._subset_names = [self._split_and_join_site(_p.name) for _p in all_patient_dirs]
            # remove duplicates
            self._subset_names = set(self._subset_names)
            # self._subset_names  = {
            #     "ms-isbi-01",
            #     "ms-isbi-02",
            #     "ms-isbi-03",
            #     "ms-isbi-04",
            #     "ms-isbi-05",
            #     "mslub",
            #     "msseg-train-center01",
            #     "msseg-test-center01",
            #     "msseg-test-center03",
            #     "msseg-train-center07",
            #     "msseg-test-center07",
            #     "msseg-train-center08",
            #     "msseg-test-center08",
            # }

        # define split for (train, val) and test
        test_centers = ("center03", "center07")
        match self.split:
            case "train" | "val":
                self._subset_names = {
                    _p for _p in self._subset_names if all(s not in _p for s in test_centers)
                }
            case "test":
                # put msseg-*-center03 in test because it GE scanner
                # and msseg-*-center07 because it is a 1.5 T scanner
                # (i.e. different from the other scanners)
                self._subset_names = {
                    _p for _p in self._subset_names if any(s in _p for s in test_centers)
                }
            case _:
                raise ValueError(f"split {self.split} not supported")

        # return as list to keep order
        return list(self._subset_names)

    def _fn2scanner(self, seq_fp: Path) -> str:
        """Get the scanner name from the filename."""
        scanner = self._split_and_join_site(seq_fp.parent.parent.name)
        return scanner

    def _split_and_join_site(self, subject_dir_name: str) -> str:
        return "-".join(subject_dir_name.split("-")[:-1])
