import pandas as pd
import numpy as np
from os.path import join
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from drfp import DrfpEncoder
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from bioservices import *
from data_preprocessing import *
import warnings

warnings.filterwarnings("ignore")

CURRENT_DIR = os.getcwd()

mol_folder = join("..", "..", "data", "metabolite_data", "mol-files")


def convert_str_to_list(str_fp):
    """
    converting string bit to list of bits
    """
    fp_list = [int(bit) for bit in str_fp]
    return fp_list


def get_reaction_site_smarts(metabolites):
    reaction_site = ""
    for met in metabolites:
        is_kegg_id = False

        if met[0] == "C":
            is_kegg_id = True

        if is_kegg_id:
            try:
                Smarts = Chem.MolToSmarts(
                    Chem.MolFromMolFile(join(mol_folder, met + ".mol"))
                )
            except OSError:
                return np.nan
        else:
            mol = Chem.inchi.MolFromInchi(met)
            if mol is not None:
                Smarts = Chem.MolToSmarts(mol)
            else:
                return np.nan
        reaction_site = reaction_site + "." + Smarts
    return reaction_site[1:]


def get_reaction_site_smiles(metabolites):
    reaction_site = ""
    for met in metabolites:
        is_kegg_id = False

        if met[0] == "C":
            is_kegg_id = True

        if is_kegg_id:
            try:
                Smarts = Chem.MolToSmiles(
                    Chem.MolFromMolFile(join(mol_folder, met + ".mol"))
                )
            except OSError:
                return np.nan
        else:
            mol = Chem.inchi.MolFromInchi(met)
            if mol is not None:
                Smarts = Chem.MolToSmiles(mol)
            else:
                return np.nan
        reaction_site = reaction_site + "." + Smarts
    return reaction_site[1:]


def convert_fp_to_array(difference_fp_dict):
    fp = np.zeros(2048)
    for key in difference_fp_dict.keys():
        fp[key] = difference_fp_dict[key]
    return fp


def create_filtered_dfs(data_df, index_ranges):
    """Function to create filtered DataFrames for train and test data"""
    filtered_dfs = {}
    for key, index_range in index_ranges.items():
        filtered_dfs[key] = data_df.iloc[:, index_range]
    return filtered_dfs


def concatenate_paac_features(df, paac_feats):
    """Function to concatenate PAAC features for train and test data"""
    paac_concat_df = pd.concat([df[paac_feats[0]], df[paac_feats[1]]], axis=1)
    # adding concatenated paac key for train and test df
    df["paac"] = paac_concat_df
    # deleting the paac feats
    for key in paac_feats:
        df.pop(key)

    return df

    #  Normalizing new-input features


def normalize_feature(df, feature_df, feature_name):
    # concatenating the features
    df[f"{feature_name}_combined"] = feature_df.apply(lambda row: row.tolist(), axis=1)
    feature_arr = np.array(df[f"{feature_name}_combined"].tolist())

    # Calculate mean and std for the current feature dataframe
    mean, std = feature_arr.mean(), feature_arr.std()

    # Apply normalization using the calculated mean and std
    df[f"{feature_name}_normalized"] = df[f"{feature_name}_combined"].apply(
        lambda x: (x - mean) / std
    )
    return df
    #  Normalizing old-input features


def get_feat_stats(df, feat):
    feat_arr = np.array(df[feat].tolist())
    mean, std = feat_arr.mean(), feat_arr.std()
    return mean, std


def normalize_features(feat_list, feat_mean, feat_std):
    norm_feat_list = (feat_list - feat_mean) / feat_std
    return norm_feat_list


#  Normalizing output feature
def calcuate_stats(feat_series):
    mean = feat_series.mean()
    std = feat_series.std()
    return mean, std


def get_processed_df(dfs_dict, df):
    # Apply normalization function for each feature for train data
    for feature_name, feature_df in dfs_dict.items():
        df = normalize_feature(df, feature_df, feature_name)
    return df


def run_step1():
    ### (a) Loading BRENDA data
    df_Brenda = pd.read_pickle(join("..", "..", "data", "kcat_data", "BRENDA_kcat.pkl"))
    # adding reaction information:
    df_Brenda.rename(columns={"correct reaction ID": "BRENDA reaction ID"})

    df_Brenda["Uniprot ID"] = np.nan
    for ind in df_Brenda.index:
        try:
            df_Brenda["Uniprot ID"][ind] = df_Brenda["UNIPROT_list"][ind][0]
        except IndexError:
            pass

    df_Brenda = df_Brenda.loc[~pd.isnull(df_Brenda["Uniprot ID"])]

    df_Brenda.drop(
        columns=[
            "index",
            "ID",
            "comment",
            "kcat",
            "kcat_new",
            "enzyme",
            "new",
            "LITERATURE",
            "UNIPROT_list",
            "new enzyme",
        ],
        inplace=True,
    )

    df_Brenda.rename(
        columns={
            "correct kcat": "kcat",
            "correct reaction ID": "BRENDA reaction ID",
            "substrate_ID_list": "substrate_IDs",
            "product_ID_list": "product_IDs",
        },
        inplace=True,
    )

    print("Number of data points: %s" % len(df_Brenda))
    print("Number of UniProt IDs: %s" % len(set(df_Brenda["Uniprot ID"])))
    print(
        "Number of checked data points: %s" % len(df_Brenda.loc[df_Brenda["checked"]])
    )
    print(
        "Number of unchecked data points: %s"
        % len(df_Brenda.loc[~df_Brenda["checked"]])
    )

    df_Brenda["from BRENDA"] = 1
    df_Brenda["from Uniprot"] = 0
    df_Brenda["from Sabio"] = 0
    df_Brenda.head()
    ### (b) Loading Sabio data
    df_Sabio = pd.read_pickle(join("..", "..", "data", "kcat_data", "Sabio_kcat.pkl"))
    df_Sabio.drop(columns=["unit", "complete", "KEGG ID"], inplace=True)
    df_Sabio.rename(columns={"products_IDs": "product_IDs"}, inplace=True)

    print("Number of data points: %s" % len(df_Sabio))
    print("Number of UniProt IDs: %s" % len(set(df_Sabio["Uniprot ID"])))

    df_Sabio["checked"] = False
    df_Sabio["#UIDs"] = 1
    df_Sabio["complete"] = True

    df_Sabio["from BRENDA"] = 0
    df_Sabio["from Uniprot"] = 0
    df_Sabio["from Sabio"] = 1
    df_Sabio.head()
    ### (c) Loading UniProt data:
    df_Uniprot = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "Uniprot_kcat.pkl")
    )

    df_Uniprot.drop(columns=["unit", "reaction ID"], inplace=True)
    df_Uniprot.rename(
        columns={
            "substrate CHEBI IDs": "Substrates",
            "product CHEBI IDs": "Products",
            "substrate InChIs": "substrate_IDs",
            "product InChIs": "product_IDs",
            "kcat [1/sec]": "kcat",
        },
        inplace=True,
    )

    print("Number of data points: %s" % len(df_Uniprot))
    print("Number of UniProt IDs: %s" % len(set(df_Uniprot["Uniprot ID"])))

    df_Uniprot["checked"] = False
    df_Uniprot["#UIDs"] = 1

    df_Uniprot["from BRENDA"] = 0
    df_Uniprot["from Uniprot"] = 1
    df_Uniprot["from Sabio"] = 0
    df_Uniprot.head()
    ### (d) Merging all three datasets
    df_kcat = pd.concat(
        [pd.concat([df_Sabio, df_Brenda], ignore_index=True), df_Uniprot],
        ignore_index=True,
    )
    df_kcat = df_kcat.loc[~pd.isnull(df_kcat["kcat"])]

    print("Number of data points: %s" % len(df_kcat))
    print("Number of UniProt IDs: %s" % len(set(df_kcat["Uniprot ID"])))
    df_kcat.to_pickle(join("..", "..", "data", "kcat_data", "kcat_data_merged.pkl"))
    df_kcat.head(2)
    ### (e) Removing duplicated entries:
    df_kcat = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "kcat_data_merged.pkl")
    )
    #### Searching for identitcal pairs of UniProt IDs and kcat values:
    droplist = []

    for ind in df_kcat.index:
        UID, kcat = df_kcat["Uniprot ID"][ind], df_kcat["kcat"][ind]
        help_df = df_kcat.loc[df_kcat["Uniprot ID"] == UID].loc[df_kcat["kcat"] == kcat]

        (
            df_kcat["from BRENDA"][ind],
            df_kcat["from Uniprot"][ind],
            df_kcat["from Sabio"][ind],
        ) = (
            max(help_df["from BRENDA"]),
            max(help_df["from Uniprot"]),
            max(help_df["from Sabio"]),
        )
        df_kcat["checked"][ind] = any(help_df["checked"])

        if len(help_df) > 1:
            droplist = droplist + list(help_df.index)[1:]
    df_kcat.drop(list(set(droplist)), inplace=True)
    print("Dropping %s data points, because they are duplicated." % len(set(droplist)))
    df_kcat.to_pickle(
        join(
            "..",
            "..",
            "data",
            "kcat_data",
            "kcat_data_step1_out.pkl",
        )
    )


# def run_step2(df_kcat):
def run_step2():
    ## 2. Downloading amino acid sequences for all data points:
    ### (a) Downloading sequences via UniProt IDs:
    # Creating a txt file with all Uniprot IDs
    df_kcat = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "kcat_data_step1_out.pkl")
    )

    IDs = list(set(df_kcat["Uniprot ID"]))
    f = open(join("..", "..", "data", "enzyme_data", "UNIPROT_IDs.txt"), "w")
    for ID in list(set(IDs)):
        f.write(str(ID) + "\n")
    f.close()
    # Mapping Uniprot IDs to sequences via the UniProt mapping service and saving the results in the file "UNIPROT_results.tab"
    UNIPROT_df = pd.read_csv(
        join("..", "..", "data", "enzyme_data", "UNIPROT_results.tab"), sep="\t"
    )
    UNIPROT_df.drop(columns=["Entry"], inplace=True)
    df_kcat = df_kcat.merge(UNIPROT_df, how="left", on="Uniprot ID")
    df_kcat = df_kcat.loc[~pd.isnull(df_kcat["Uniprot ID"])]
    print(
        "Number of different amino acid sequences in the dataset: %s"
        % len(set(df_kcat["Sequence"]))
    )
    df_kcat.to_pickle(
        join(
            "..",
            "..",
            "data",
            "kcat_data",
            "kcat_data_step2_out.pkl",
        )
    )


def run_step3():
    ## 3. Mapping all substrates and products to InChI strings:
    #### Most of the metabolites in our dataset have InChI strings as identifiers and some of them have KEGG Compound IDs. We are trying to map the KEGG Compound IDs to InChI strings as well:
    #### (a) Getting an InChI string for all metabolites
    df_kcat = pd.read_pickle("../../data/kcat_data/kcat_data_step2_out.pkl")
    kegg_con = KEGG()
    chebi_con = ChEBI()
    met_IDs = []

    for ind in df_kcat.index:
        sub_IDs, pro_IDs = df_kcat["substrate_IDs"][ind], df_kcat["product_IDs"][ind]
        if sub_IDs != "" and pro_IDs != "":
            try:
                met_IDs = met_IDs + sub_IDs + pro_IDs
            except TypeError:
                pass

    df_metabolites = pd.DataFrame(data={"metabolite ID": list(set(met_IDs))})
    df_metabolites = df_metabolites.loc[df_metabolites["metabolite ID"] != ""]
    df_metabolites["InChI"] = np.nan

    for ind in df_metabolites.index:
        met = df_metabolites["metabolite ID"][ind]
        if met[0:5] == "InChI":
            df_metabolites["InChI"][ind] = met
        else:
            try:
                kegg_entry = kegg_con.parse(kegg_con.get(met))
                chebi_entry = chebi_con.getCompleteEntity(
                    "CHEBI:" + kegg_entry["DBLINKS"]["ChEBI"]
                )
                df_metabolites["InChI"][ind] = chebi_entry.inchi
            except:
                pass

    df_metabolites.head()
    from os.path import join

    for ind in df_metabolites.index:
        if pd.isnull(df_metabolites["InChI"][ind]):
            try:
                mol = Chem.MolFromMolFile(
                    join(
                        "..",
                        "..",
                        "data",
                        "metabolite_data",
                        "mol-files",
                        df_metabolites["metabolite ID"][ind] + ".mol",
                    )
                )
                df_metabolites["InChI"][ind] = Chem.MolToInchi(mol)
            except:
                pass

    df_metabolites = df_metabolites.loc[~pd.isnull(df_metabolites["InChI"])]
    #### (b) Mapping the InChI strings for all substrates and all products to the kcat values:
    df_kcat["substrate_InChI_set"] = ""
    df_kcat["product_InChI_set"] = ""

    for ind in df_kcat.index:
        sub_IDs, pro_IDs = df_kcat["substrate_IDs"][ind], df_kcat["product_IDs"][ind]

        try:
            sub_inchis = []
            pro_inchis = []
            for sub in sub_IDs:
                inchi = list(
                    df_metabolites["InChI"].loc[df_metabolites["metabolite ID"] == sub]
                )[0]
                sub_inchis.append(inchi)
            for pro in pro_IDs:
                inchi = list(
                    df_metabolites["InChI"].loc[df_metabolites["metabolite ID"] == pro]
                )[0]
                pro_inchis.append(inchi)

            df_kcat["substrate_InChI_set"][ind] = set(sub_inchis)
            df_kcat["product_InChI_set"][ind] = set(pro_inchis)
        except:
            pass

    df_kcat.to_pickle(
        join(
            "..",
            "..",
            "data",
            "kcat_data",
            "kcat_data_step3_out.pkl",
        )
    )


def run_step4():
    ## 4. Assigning  IDs to every unique sequence and to every unique reaction in the dataset:
    df_kcat = pd.read_pickle("../../data/kcat_data/kcat_data_step3_out.pkl")
    #### (a) Creating DataFrames for all sequences and for all reactions:
    df_sequences = pd.DataFrame(data={"Sequence": list(set(df_kcat["Sequence"]))})
    df_sequences = df_sequences.loc[~pd.isnull(df_sequences["Sequence"])]
    df_sequences.reset_index(inplace=True, drop=True)
    df_sequences["Sequence ID"] = ["Sequence_" + str(ind) for ind in df_sequences.index]

    df_sequences
    df_reactions = pd.DataFrame(
        {
            "substrates": df_kcat["substrate_InChI_set"],
            "products": df_kcat["product_InChI_set"],
        }
    )
    df_reactions = df_reactions.loc[df_reactions["substrates"] != set([])]
    df_reactions = df_reactions.loc[df_reactions["products"] != set([])]

    droplist = []
    for ind in df_reactions.index:
        sub_IDs, pro_IDs = (
            df_reactions["substrates"][ind],
            df_reactions["products"][ind],
        )
        help_df = df_reactions.loc[df_reactions["substrates"] == sub_IDs].loc[
            df_reactions["products"] == pro_IDs
        ]
        if len(help_df):
            for ind in list(help_df.index)[1:]:
                droplist.append(ind)

    df_reactions.drop(list(set(droplist)), inplace=True)
    df_reactions.reset_index(inplace=True, drop=True)

    df_reactions["Reaction ID"] = ["Reaction_" + str(ind) for ind in df_reactions.index]
    df_reactions
    #### (b) Calcuating the sum of the molecular weights of all substrates and of all products:
    df_reactions["MW_frac"] = np.nan

    for ind in df_reactions.index:
        substrates = list(df_reactions["substrates"][ind])
        products = list(df_reactions["products"][ind])

        mw_subs = mw_mets(metabolites=substrates)
        mw_pros = mw_mets(metabolites=products)
        if mw_pros != 0:
            df_reactions["MW_frac"][ind] = mw_subs / mw_pros
        else:
            df_reactions["MW_frac"][ind] = np.inf

    df_reactions
    #### (c) Mapping Sequence and Reaction IDs to kcat_df:
    df_kcat = df_kcat.merge(df_sequences, on="Sequence", how="left")
    df_reactions.rename(
        columns={"substrates": "substrate_InChI_set", "products": "product_InChI_set"},
        inplace=True,
    )

    df_kcat["Reaction ID"] = np.nan
    df_kcat["MW_frac"] = np.nan
    for ind in df_kcat.index:
        sub_set, pro_set = (
            df_kcat["substrate_InChI_set"][ind],
            df_kcat["product_InChI_set"][ind],
        )

        help_df = df_reactions.loc[df_reactions["substrate_InChI_set"] == sub_set].loc[
            df_reactions["product_InChI_set"] == pro_set
        ]
        if len(help_df) == 1:
            df_kcat["Reaction ID"][ind] = list(help_df["Reaction ID"])[0]
            df_kcat["MW_frac"][ind] = list(help_df["MW_frac"])[0]
    df_kcat.head(2)
    #### (d) Creating a new DataFrame with one entry for every unique sequence-reaction pair:
    ##### (d)(i) Creating the DataFrame:
    df_kcat_new = pd.DataFrame(
        data={
            "Reaction ID": df_kcat["Reaction ID"],
            "Sequence ID": df_kcat["Sequence ID"],
        }
    )
    df_kcat_new = df_kcat_new.loc[~pd.isnull(df_kcat_new["Reaction ID"])].loc[
        ~pd.isnull(df_kcat_new["Sequence ID"])
    ]
    df_kcat_new.drop_duplicates(inplace=True)
    df_kcat_new.reset_index(inplace=True, drop=True)

    df_kcat_new["kcat_values"], df_kcat_new["Uniprot IDs"] = "", ""
    (
        df_kcat_new["from_BRENDA"],
        df_kcat_new["from_Sabio"],
        df_kcat_new["from_Uniprot"],
    ) = ("", "", "")
    df_kcat_new["checked"] = ""

    for ind in df_kcat_new.index:
        RID, SID = df_kcat_new["Reaction ID"][ind], df_kcat_new["Sequence ID"][ind]
        help_df = df_kcat.loc[df_kcat["Reaction ID"] == RID].loc[
            df_kcat["Sequence ID"] == SID
        ]

        df_kcat_new["kcat_values"][ind] = list(help_df["kcat"])
        df_kcat_new["Uniprot IDs"][ind] = list(help_df["Uniprot ID"])
        df_kcat_new["from_BRENDA"][ind] = list(help_df["from BRENDA"])
        df_kcat_new["from_Sabio"][ind] = list(help_df["from Sabio"])
        df_kcat_new["from_Uniprot"][ind] = list(help_df["from Uniprot"])
        df_kcat_new["checked"][ind] = list(help_df["checked"])
    df_kcat_new
    ##### (d)(ii): Adding sequence, substrates, and products to all data points
    (
        df_kcat_new["Sequence"],
        df_kcat_new["substrates"],
        df_kcat_new["products"],
        df_kcat_new["MW_frac"],
    ) = ("", "", "", "")

    for ind in df_kcat_new.index:
        RID, SID = df_kcat_new["Reaction ID"][ind], df_kcat_new["Sequence ID"][ind]
        help_df = df_reactions.loc[df_reactions["Reaction ID"] == RID]
        df_kcat_new["substrates"][ind], df_kcat_new["products"][ind] = (
            list(help_df["substrate_InChI_set"])[0],
            list(help_df["product_InChI_set"])[0],
        )
        df_kcat_new["MW_frac"][ind] = list(help_df["MW_frac"])[0]

        help_df = df_sequences.loc[df_sequences["Sequence ID"] == SID]
        df_kcat_new["Sequence"][ind] = list(help_df["Sequence"])[0]

    ##### (d)(iii) Calculating the maximal kcat value for every sequence and for every reaction:
    df_all_kcat = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "kcat_data_merged.pkl")
    )
    df_all_kcat.head()

    df_kcat_new["max_kcat_for_UID"] = ""
    df_kcat_new["max_kcat_for_RID"] = ""

    for ind in df_kcat_new.index:
        max_kcat = -np.inf
        UIDs = list(set(df_kcat_new["Uniprot IDs"][ind]))
        for UID in UIDs:
            all_kcat = list(df_all_kcat["kcat"].loc[df_all_kcat["Uniprot ID"] == UID])
            all_kcat = [float(kcat) for kcat in all_kcat]
            max_kcat = max(max_kcat, max(all_kcat))
        df_kcat_new["max_kcat_for_UID"][ind] = max_kcat

    for ind in df_kcat_new.index:
        RID = df_kcat_new["Reaction ID"][ind]

        help_df = df_kcat_new.loc[df_kcat_new["Reaction ID"] == RID]
        all_kcat = []
        for ind2 in help_df.index:
            all_kcat = all_kcat + list(help_df["kcat_values"][ind2])
        all_kcat = [float(kcat) for kcat in all_kcat]
        max_kcat = max(all_kcat)
        df_kcat_new["max_kcat_for_RID"][ind] = max_kcat
    df_kcat_new.head()
    ##### (d)(iv) Calculating the maximal kcat value for every EC number in the dataset:
    df_kcat = df_kcat_new.copy()

    # Using the txt file and the Uniprot mapping service to get an EC number for every enzyme:
    df_EC = pd.read_csv(
        join("..", "..", "data", "enzyme_data", "Uniprot_results_EC.tab"), sep="\t"
    )
    df_EC.head()
    df_kcat.head()
    df_kcat["ECs"] = ""
    for ind in df_kcat.index:
        UID = df_kcat["Uniprot IDs"][ind][0]
        try:
            df_kcat["ECs"][ind] = list(
                df_EC["EC number"].loc[df_EC["Uniprot ID"] == UID]
            )[0].split("; ")
        except:
            df_kcat["ECs"][ind] = []
    df_kcat.head(2)
    all_ECs = []
    for ind in df_kcat.index:
        all_ECs = all_ECs + df_kcat["ECs"][ind]

    all_ECs = list(set(all_ECs))

    df_EC_kcat = pd.DataFrame({"EC": all_ECs})
    df_EC_kcat["max_kcat"] = np.nan

    for ind in df_EC_kcat.index:
        try:
            kcat_max = get_max_for_EC_number(EC=df_EC_kcat["EC"][ind])
            df_EC_kcat["max_kcat"][ind] = kcat_max
            print(ind, kcat_max)
        except:
            pass

    df_EC_kcat.to_pickle(join("..", "..", "data", "enzyme_data", "df_EC_max_kcat.pkl"))
    # Mapping max EC kcat value to all data points:
    df_EC_kcat = pd.read_pickle(
        join("..", "..", "data", "enzyme_data", "df_EC_max_kcat.pkl")
    )
    df_kcat["max_kcat_for_EC"] = np.nan

    for ind in df_kcat.index:
        ECs = df_kcat["ECs"][ind]
        max_kcat = 0
        for EC in ECs:
            try:
                max_kcat = max(
                    max_kcat,
                    list(df_EC_kcat["max_kcat"].loc[df_EC_kcat["EC"] == EC])[0],
                )
            except:
                pass
        if max_kcat != 0:
            df_kcat["max_kcat_for_EC"][ind] = max_kcat
    df_kcat.to_pickle(
        join("..", "..", "data", "kcat_data", "merged_and_grouped_kcat_dataset.pkl")
    )
    df_sequences.to_pickle(
        join("..", "..", "data", "enzyme_data", "all_sequences_with_IDs.pkl")
    )
    df_reactions.to_pickle(
        join("..", "..", "data", "reaction_data", "all_reactions_with_IDs.pkl")
    )


def run_step5():
    ## 5. Calculating reaction fingerprints (structural and difference) for every reaction and a ESM-1b/ESM-1b_ts vector for every amino acid sequence:
    #### (a) Executing jupyter notebook A2 to calculate the reaction fingerprints and enzyme representations. Then loading the results
    df_kcat = pd.read_pickle(join("..", "..", "data", "kcat_data", "new_features.pkl"))
    # df_kcat = pd.read_pickle(join("..", "..", "data", "kcat_data", "merged_and_grouped_kcat_dataset.pkl"))

    df_sequences = pd.read_pickle(
        join(
            "..", "..", "data", "enzyme_data", "all_sequences_with_IDs_and_ESM1b_ts.pkl"
        )
    )
    df_reactions = pd.read_pickle(
        join("..", "..", "data", "reaction_data", "all_reactions_with_IDs_and_FPs.pkl")
    )

    df_sequences = df_sequences[["Sequence", "ESM1b", "ESM1b_ts"]]
    len(df_reactions), len(df_sequences)
    #### (b) Mapping ESM-1b vectors and reaction fingerprints to kcat dataset:
    # Mapping Enzyme Sequence to Numeric feature representation
    merged_sequences_df = pd.merge(df_kcat, df_sequences, on="Sequence")
    merged_sequences_df

    # Finding fingerprints (DRFP, Structural, difference) vectors for each substrate and product of chemical reaction
    (
        merged_sequences_df["structural_fp"],
        merged_sequences_df["difference_fp"],
        merged_sequences_df["DRFP"],
    ) = ("", "", "")
    merged_sequences_df["#substrates"], merged_sequences_df["#products"] = "", ""

    for ind in merged_sequences_df.index:
        substrates = list(merged_sequences_df["substrates"][ind])
        products = list(merged_sequences_df["products"][ind])
        try:
            left_site = get_reaction_site_smarts(substrates)
            right_site = get_reaction_site_smarts(products)
            if not pd.isnull(left_site) and not pd.isnull(right_site):

                rxn_forward = AllChem.ReactionFromSmarts(left_site + ">>" + right_site)

                difference_fp = (
                    Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(
                        rxn_forward
                    )
                )
                difference_fp = convert_fp_to_array(difference_fp.GetNonzeroElements())
                structural_fp = (
                    Chem.rdChemReactions.CreateStructuralFingerprintForReaction(
                        rxn_forward
                    ).ToBitString()
                )

                left_site = get_reaction_site_smiles(substrates)
                right_site = get_reaction_site_smiles(products)
                drfp = DrfpEncoder.encode(left_site + ">>" + right_site)[0]

                merged_sequences_df["DRFP"][ind] = drfp
                merged_sequences_df["structural_fp"][ind] = structural_fp
                merged_sequences_df["difference_fp"][ind] = difference_fp
                merged_sequences_df["#substrates"][ind] = len(substrates)
                merged_sequences_df["#products"][ind] = len(products)
        except IndexError:
            pass

    df_kcat = merged_sequences_df.copy()
    n = len(df_kcat)
    # Remove values with missing reaction fingerprints or enzyme representation
    df_kcat = (
        df_kcat.loc[df_kcat["structural_fp"] != ""]
        .loc[df_kcat["ESM1b"] != ""]
        .loc[df_kcat["ESM1b_ts"] != ""]
    )
    # df_kcat = df_kcat.loc[df_kcat["structural_fp"] != ""].loc[df_kcat["ESM1b"].str.len() > 0].loc[df_kcat["ESM1b_ts"].str.len() > 0]
    print(
        "Removing %s enzyme-reaction combinations because they either do not have a ESM1b vector or reaction fingerprint"
        % (n - len(df_kcat))
    )

    df_kcat["structural_fp"] = df_kcat["structural_fp"].apply(convert_str_to_list)
    df_kcat.to_pickle(
        join(
            "..",
            "..",
            "data",
            "kcat_data",
            "merged_and_grouped_kcat_dataset_with_FPs_and_ESM1bs_ts.pkl",
        )
    )


def run_step6():
    ## 6.Removing outliers and non-natural reactions:
    df_kcat = pd.read_pickle(
        join(
            "..",
            "..",
            "data",
            "kcat_data",
            "merged_and_grouped_kcat_dataset_with_FPs_and_ESM1bs_ts.pkl",
        )
    )
    #### (a) Calculating the geometric mean and log10-transforming it for all enzyme-reaction pairs:
    # To ignore $k_{cat}$ values that were obtained under non-optimal conditions, we exclude values lower than 1\% than the maximal $k_{cat}$ value for the same enzyme-reaction combination.
    df_kcat["geomean_kcat"] = np.nan
    df_kcat["frac_of_max_UID"] = np.nan
    df_kcat["frac_of_max_RID"] = np.nan
    df_kcat["frac_of_max_EC"] = np.nan

    for ind in df_kcat.index:
        all_kcat = np.array(df_kcat["kcat_values"][ind]).astype(float)
        max_kcat = max(all_kcat)
        all_kcat_top = [kcat for kcat in all_kcat if kcat / max_kcat > 0.01]
        df_kcat["geomean_kcat"][ind] = np.mean(np.log10(all_kcat_top))

        df_kcat["frac_of_max_UID"][ind] = (
            np.max(np.array(df_kcat["kcat_values"][ind]).astype(float))
            / df_kcat["max_kcat_for_UID"][ind]
        )
        df_kcat["frac_of_max_RID"][ind] = (
            np.max(np.array(df_kcat["kcat_values"][ind]).astype(float))
            / df_kcat["max_kcat_for_RID"][ind]
        )
        df_kcat["frac_of_max_EC"][ind] = (
            np.max(np.array(df_kcat["kcat_values"][ind]).astype(float))
            / df_kcat["max_kcat_for_EC"][ind]
        )
    df_kcat = df_kcat.loc[~pd.isnull(df_kcat["geomean_kcat"])]

    len(df_kcat)
    #### (b) We are only interested in kcat values that were measured for the natural reaction of an enzyme:
    # To achieve this we exclude kcat values for an enzyme if another measurement exists for the same enzyme but for different reaction with a kcat value that is more than ten times higher. Furthermore, to exlcude data points measured under non-optimal conditions and for non-natural reactions, we exclude kcat values if we could find a kcat value for the same reaction or same EC number that is more than 100 times higher.
    n = len(df_kcat)

    df_kcat = df_kcat.loc[df_kcat["frac_of_max_UID"] > 0.1]
    df_kcat = df_kcat.loc[df_kcat["frac_of_max_RID"] > 0.01]

    df_kcat["frac_of_max_EC"].loc[pd.isnull(df_kcat["frac_of_max_EC"])] = 1
    df_kcat = df_kcat.loc[df_kcat["frac_of_max_EC"] < 10]
    df_kcat = df_kcat.loc[df_kcat["frac_of_max_EC"] > 0.01]
    print(
        "We remove %s data points, because we suspect that these kcat values were not measure for the natural reaction "
        "of an enzyme or under non-optimal conditions." % (n - len(df_kcat))
    )
    #### (c) Removing data points with reaction queations with uneven fraction of molecular weights
    n = len(df_kcat)

    df_kcat = df_kcat.loc[df_kcat["MW_frac"] < 3]
    df_kcat = df_kcat.loc[df_kcat["MW_frac"] > 1 / 3]

    print(
        "We remove %s data points because the sum of molecular weights of substrates does not match the sum of molecular"
        "weights of the products." % (n - len(df_kcat))
    )
    #### (d) Removing data points with outlying kcat values:
    n = len(df_kcat)

    df_kcat = df_kcat.loc[~(df_kcat["geomean_kcat"] > 5)]
    df_kcat = df_kcat.loc[~(df_kcat["geomean_kcat"] < -2.5)]

    print(
        "We remove %s data point because their kcat values are outliers."
        % (n - len(df_kcat))
    )
    df_kcat.dropna(axis=1, inplace=True)
    print("Size of final kcat dataset: %s" % len(df_kcat))
    df_kcat.to_pickle(join("..", "..", "data", "kcat_data", "final_kcat_dataset.pkl"))


def run_step7():
    ## 7. Splitting the dataset into training and test set:
    #### (a) Splitting the dataset in such a way that the same enzyme does not occur in the training and the test set:
    df_kcat = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "final_kcat_dataset.pkl")
    )
    df_kcat.head()
    # Shuffling DataFrame:
    df = df_kcat.copy()
    df = df.sample(frac=1, random_state=123)
    df.reset_index(drop=True, inplace=True)
    # Splitting dataset
    train_df, test_df = split_dataframe_enzyme(frac=5, df=df.copy())
    print("Test set size: %s" % len(test_df))
    print("Training set size: %s" % len(train_df))
    print(
        "Size of test set in percent: %s"
        % np.round(100 * len(test_df) / (len(test_df) + len(train_df)))
    )

    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    train_df.to_pickle(
        join("..", "..", "data", "kcat_data", "splits", "train_df_kcat_new.pkl")
    )
    test_df.to_pickle(
        join("..", "..", "data", "kcat_data", "splits", "test_df_kcat_new.pkl")
    )
    #### (b) Splitting the training set into 5 folds for 5-fold cross-validations (CVs):
    # In order to achieve a model that generalizes well during CV, we created the 5 folds in such a way that neither the same enzyme nor the same reaction occurs in two different subsets.
    train_df = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "splits", "train_df_kcat_new.pkl")
    )
    test_df = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "splits", "test_df_kcat_new.pkl")
    )
    data_train2 = train_df.copy()
    data_train2["index"] = list(data_train2.index)

    data_train2, df_fold = split_dataframe_enzyme(df=data_train2, frac=5)
    indices_fold1 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold1))  #

    data_train2, df_fold = split_dataframe_enzyme(df=data_train2, frac=4)
    indices_fold2 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold2))

    data_train2, df_fold = split_dataframe_enzyme(df=data_train2, frac=3)
    indices_fold3 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold3))

    data_train2, df_fold = split_dataframe_enzyme(df=data_train2, frac=2)
    indices_fold4 = list(df_fold["index"])
    indices_fold5 = list(data_train2["index"])
    print(len(data_train2), len(indices_fold4))

    fold_indices = [
        indices_fold1,
        indices_fold2,
        indices_fold3,
        indices_fold4,
        indices_fold5,
    ]

    train_indices = [[], [], [], [], []]
    test_indices = [[], [], [], [], []]

    for i in range(5):
        for j in range(5):
            if i != j:
                train_indices[i] = train_indices[i] + fold_indices[j]
        test_indices[i] = fold_indices[i]

    np.save(
        join("..", "..", "data", "kcat_data", "splits", "CV_train_indices"),
        train_indices,
    )
    np.save(
        join("..", "..", "data", "kcat_data", "splits", "CV_test_indices"), test_indices
    )


def run_step8():
    train_df = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "splits", "train_df_kcat_new.pkl")
    )
    test_df = pd.read_pickle(
        join("..", "..", "data", "kcat_data", "splits", "test_df_kcat_new.pkl")
    )
    index_ranges = {
        "maccs_keys": slice(15, 182),
        "chemical_descriptors": slice(182, 372),
        "aa_composition": slice(372, 392),
        "dp_composition": slice(392, 792),
        "moreau_broto_auto": slice(792, 1032),
        "moran_auto": slice(1032, 1272),
        "geary_auto": slice(1272, 1512),
        "ctd": slice(1512, 1659),
        "paac_1": slice(1659, 1689),
        "apaac": slice(1689, 1719),
        "paac_2": slice(1719, 1729),
        "socn": slice(1729, 1819),
        "qso": slice(1819, 1919),
        "traid": slice(1919, 2262),
    }

    train_dfs = create_filtered_dfs(train_df, index_ranges)
    test_dfs = create_filtered_dfs(test_df, index_ranges)

    paac_feats = ["paac_1", "paac_2"]
    train_dfs = concatenate_paac_features(train_dfs, paac_feats)
    test_dfs = concatenate_paac_features(test_dfs, paac_feats)

    train_dfs.keys(), test_dfs.keys()

    processed_train_df = get_processed_df(train_dfs, train_df)
    processed_test_df = get_processed_df(test_dfs, test_df)

    # train dataset
    mean, std = get_feat_stats(processed_train_df, "ESM1b")
    processed_train_df["ESM1b_norm"] = processed_train_df["ESM1b"].apply(
        lambda x: normalize_features(x, mean, std)
    )

    mean, std = get_feat_stats(processed_train_df, "ESM1b_ts")
    processed_train_df["ESM1b_ts_norm"] = processed_train_df["ESM1b_ts"].apply(
        lambda x: normalize_features(x, mean, std)
    )

    # test dataset
    mean, std = get_feat_stats(processed_test_df, "ESM1b")
    processed_test_df["ESM1b_norm"] = processed_test_df["ESM1b"].apply(
        lambda x: normalize_features(x, mean, std)
    )

    mean, std = get_feat_stats(processed_test_df, "ESM1b_ts")
    processed_test_df["ESM1b_ts_norm"] = processed_test_df["ESM1b_ts"].apply(
        lambda x: normalize_features(x, mean, std)
    )

    mean, std = calcuate_stats(processed_train_df["geomean_kcat"])
    processed_train_df["log10_kcat_norm"] = processed_train_df["geomean_kcat"].apply(
        lambda x: (x - mean) / std
    )

    mean, std = calcuate_stats(processed_test_df["geomean_kcat"])
    processed_test_df["log10_kcat_norm"] = processed_test_df["geomean_kcat"].apply(
        lambda x: (x - mean) / std
    )

    processed_test_df.to_pickle(
        join("..", "..", "data", "kcat_data", "splits", "test_df_kcat_new_feats.pkl")
    )
    processed_train_df.to_pickle(
        join("..", "..", "data", "kcat_data", "splits", "train_df_kcat_new_feats.pkl")
    )
