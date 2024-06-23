import pandas as pd
from PyBioMed import Pyprotein
from os.path import join
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
fpath = "D:/Sahar/Work/Enzymes/Turnover/"
AAcomp, DPcomp, TPcomp, MBA, MA, GA, CTD, PAAC, APAAC, SOCN, QSO, traid, \
macs_keys, chemical_descriptors = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), \
pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), \
pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def calc_AAcomp(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetAAComp())
    return seq_rep

def calc_DPcomp(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetDPComp())
    return seq_rep

def calc_MBA(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetMoreauBrotoAuto())
    return seq_rep

def calc_MA(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetMoranAuto())
    return seq_rep

def calc_GA(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetGearyAuto())
    return seq_rep

def calc_CTD(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetCTD())
    return seq_rep

def calc_PAAC(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetPAAC())
    return seq_rep

def calc_APAAC(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetAPAAC())
    return seq_rep

def calc_SOCN(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetSOCN())
    return seq_rep

def calc_QSO(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetQSO())
    return seq_rep

def calc_traid(seq):
    seq_rep = {}
    proclass = Pyprotein.PyProtein(seq)
    seq_rep.update(proclass.GetTriad())
    return seq_rep
    
def calc_maccs(smi):
    m = Chem.MolFromSmiles(smi)
    fps_macs1 = [*MACCSkeys.GenMACCSKeys(m).ToBitString()]
    return fps_macs1

def calc_chems(smi):
    m = Chem.MolFromSmiles(smi)
    chem_desc = np.zeros(190)
    chem_desc[0] = Descriptors.MaxEStateIndex(m)
    chem_desc[1] = Descriptors.MinEStateIndex(m)
    chem_desc[2] = Descriptors.MaxAbsEStateIndex(m)
    chem_desc[3] = Descriptors.MinAbsEStateIndex(m)
    chem_desc[4] = Descriptors.BalabanJ(m)
    chem_desc[5] = Descriptors.BertzCT(m)
    chem_desc[6] = Descriptors.Chi0(m)
    chem_desc[7] = Descriptors.Chi0n(m)
    chem_desc[8] = Descriptors.Chi0v(m)
    chem_desc[9] = Descriptors.Chi1(m)
    chem_desc[10] = Descriptors.Chi1n(m)
    chem_desc[11] = Descriptors.Chi1v(m)
    chem_desc[12] = Descriptors.Chi2n(m)
    chem_desc[13] = Descriptors.Chi2v(m)
    chem_desc[14] = Descriptors.Chi3n(m)
    chem_desc[15] = Descriptors.Chi3v(m)
    chem_desc[16] = Descriptors.Chi4n(m)
    chem_desc[17] = Descriptors.Chi4v(m)
    chem_desc[18] = Descriptors.EState_VSA1(m)
    chem_desc[19] = Descriptors.EState_VSA2(m)
    chem_desc[20] = Descriptors.EState_VSA3(m)
    chem_desc[21] = Descriptors.EState_VSA4(m)
    chem_desc[22] = Descriptors.EState_VSA5(m)
    chem_desc[23] = Descriptors.EState_VSA6(m)
    chem_desc[24] = Descriptors.EState_VSA7(m)
    chem_desc[25] = Descriptors.EState_VSA8(m)
    chem_desc[26] = Descriptors.EState_VSA9(m)
    chem_desc[27] = Descriptors.EState_VSA10(m)
    chem_desc[28] = Descriptors.EState_VSA11(m)
    chem_desc[29] = Descriptors.FractionCSP3(m)
    chem_desc[30] = Descriptors.HallKierAlpha(m)
    chem_desc[31] = Descriptors.HeavyAtomCount(m)
    chem_desc[32] = Descriptors.Ipc(m)
    chem_desc[33] = Descriptors.Kappa1(m)
    chem_desc[34] = Descriptors.Kappa2(m)
    chem_desc[35] = Descriptors.Kappa3(m)
    chem_desc[36] = Descriptors.LabuteASA(m)
    chem_desc[37] = Descriptors.MolLogP(m)
    chem_desc[38] = Descriptors.MolMR(m)
    chem_desc[39] = Descriptors.NHOHCount(m)
    chem_desc[40] = Descriptors.NOCount(m)
    chem_desc[41] = Descriptors.NumAliphaticCarbocycles(m)
    chem_desc[42] = Descriptors.NumAliphaticHeterocycles(m)
    chem_desc[43] = Descriptors.NumAliphaticRings(m)
    chem_desc[44] = Descriptors.NumAromaticCarbocycles(m)
    chem_desc[45] = Descriptors.NumAromaticHeterocycles(m)
    chem_desc[46] = Descriptors.NumAromaticRings(m)
    chem_desc[47] = Descriptors.NumHAcceptors(m)
    chem_desc[48] = Descriptors.NumHDonors(m)
    chem_desc[49] = Descriptors.NumHeteroatoms(m)
    chem_desc[50] = Descriptors.NumRotatableBonds(m)
    chem_desc[51] = Descriptors.NumSaturatedCarbocycles(m)
    chem_desc[52] = Descriptors.NumSaturatedHeterocycles(m)
    chem_desc[53] = Descriptors.PEOE_VSA1(m)
    chem_desc[54] = Descriptors.PEOE_VSA10(m)
    chem_desc[55] = Descriptors.PEOE_VSA11(m)
    chem_desc[56] = Descriptors.PEOE_VSA12(m)
    chem_desc[57] = Descriptors.PEOE_VSA13(m)
    chem_desc[58] = Descriptors.PEOE_VSA14(m)
    chem_desc[59] = Descriptors.PEOE_VSA2(m)
    chem_desc[60] = Descriptors.PEOE_VSA3(m)
    chem_desc[61] = Descriptors.PEOE_VSA4(m)
    chem_desc[62] = Descriptors.PEOE_VSA5(m)
    chem_desc[63] = Descriptors.PEOE_VSA6(m)
    chem_desc[64] = Descriptors.PEOE_VSA7(m)
    chem_desc[65] = Descriptors.PEOE_VSA8(m)
    chem_desc[66] = Descriptors.PEOE_VSA9(m)
    chem_desc[67] = Descriptors.RingCount(m)
    chem_desc[68] = Descriptors.SMR_VSA1(m)
    chem_desc[69] = Descriptors.SMR_VSA10(m)
    chem_desc[70] = Descriptors.SMR_VSA2(m)
    chem_desc[71] = Descriptors.SMR_VSA3(m)
    chem_desc[72] = Descriptors.SMR_VSA4(m)
    chem_desc[73] = Descriptors.SMR_VSA5(m)
    chem_desc[74] = Descriptors.SMR_VSA6(m)
    chem_desc[75] = Descriptors.SMR_VSA7(m)
    chem_desc[76] = Descriptors.SMR_VSA8(m)
    chem_desc[77] = Descriptors.SMR_VSA9(m)
    chem_desc[78] = Descriptors.SlogP_VSA1(m)
    chem_desc[79] = Descriptors.SlogP_VSA10(m)
    chem_desc[80] = Descriptors.SlogP_VSA11(m)
    chem_desc[81] = Descriptors.SlogP_VSA12(m)
    chem_desc[82] = Descriptors.SlogP_VSA2(m)
    chem_desc[83] = Descriptors.SlogP_VSA3(m)
    chem_desc[84] = Descriptors.SlogP_VSA4(m)
    chem_desc[85] = Descriptors.SlogP_VSA5(m)
    chem_desc[86] = Descriptors.SlogP_VSA6(m)
    chem_desc[87] = Descriptors.SlogP_VSA7(m)
    chem_desc[88] = Descriptors.SlogP_VSA8(m)
    chem_desc[89] = Descriptors.SlogP_VSA9(m)
    chem_desc[90] = Descriptors.TPSA(m)
    chem_desc[91] = Descriptors.VSA_EState1(m)
    chem_desc[92] = Descriptors.VSA_EState10(m)
    chem_desc[93] = Descriptors.VSA_EState2(m)
    chem_desc[94] = Descriptors.VSA_EState3(m)
    chem_desc[95] = Descriptors.VSA_EState4(m)
    chem_desc[96] = Descriptors.VSA_EState5(m)
    chem_desc[97] = Descriptors.VSA_EState6(m)
    chem_desc[98] = Descriptors.VSA_EState7(m)
    chem_desc[99] = Descriptors.VSA_EState8(m)
    chem_desc[100] = Descriptors.VSA_EState9(m)
    chem_desc[101] = Descriptors.fr_Al_COO(m)
    chem_desc[102] = Descriptors.fr_Al_OH(m)
    chem_desc[103] = Descriptors.fr_Al_OH_noTert(m)
    chem_desc[104] = Descriptors.fr_ArN(m)
    chem_desc[105] = Descriptors.fr_Ar_COO(m)
    chem_desc[106] = Descriptors.fr_Ar_N(m)
    chem_desc[107] = Descriptors.fr_Ar_NH(m)
    chem_desc[108] = Descriptors.fr_Ar_OH(m)
    chem_desc[109] = Descriptors.fr_COO(m)
    chem_desc[110] = Descriptors.fr_COO2(m)
    chem_desc[111] = Descriptors.fr_C_O(m)
    chem_desc[112] = Descriptors.fr_C_O_noCOO(m)
    chem_desc[113] = Descriptors.fr_C_S(m)
    chem_desc[114] = Descriptors.fr_HOCCN(m)
    chem_desc[115] = Descriptors.fr_Imine(m)
    chem_desc[116] = Descriptors.fr_NH0(m)
    chem_desc[117] = Descriptors.fr_NH1(m)
    chem_desc[118] = Descriptors.fr_NH2(m)
    chem_desc[119] = Descriptors.fr_N_O(m)
    chem_desc[120] = Descriptors.fr_Ndealkylation1(m)
    chem_desc[121] = Descriptors.fr_Ndealkylation2(m)
    chem_desc[122] = Descriptors.fr_Nhpyrrole(m)
    chem_desc[123] = Descriptors.fr_SH(m)
    chem_desc[124] = Descriptors.fr_aldehyde(m)
    chem_desc[125] = Descriptors.fr_alkyl_carbamate(m)
    chem_desc[126] = Descriptors.fr_alkyl_halide(m)
    chem_desc[127] = Descriptors.fr_allylic_oxid(m)
    chem_desc[128] = Descriptors.fr_amide(m)
    chem_desc[129] = Descriptors.fr_amidine(m)
    chem_desc[130] = Descriptors.fr_aniline(m)
    chem_desc[131] = Descriptors.fr_aryl_methyl(m)
    chem_desc[132] = Descriptors.fr_azide(m)
    chem_desc[133] = Descriptors.fr_azo(m)
    chem_desc[134] = Descriptors.fr_barbitur(m)
    chem_desc[135] = Descriptors.fr_benzene(m)
    chem_desc[136] = Descriptors.fr_benzodiazepine(m)
    chem_desc[137] = Descriptors.fr_bicyclic(m)
    chem_desc[138] = Descriptors.fr_diazo(m)
    chem_desc[139] = Descriptors.fr_dihydropyridine(m)
    chem_desc[140] = Descriptors.fr_epoxide(m)
    chem_desc[141] = Descriptors.fr_ester(m)
    chem_desc[142] = Descriptors.fr_ether(m)
    chem_desc[143] = Descriptors.fr_furan(m)
    chem_desc[144] = Descriptors.fr_guanido(m)
    chem_desc[145] = Descriptors.fr_halogen(m)
    chem_desc[146] = Descriptors.fr_hdrzine(m)
    chem_desc[147] = Descriptors.fr_hdrzone(m)
    chem_desc[148] = Descriptors.fr_imidazole(m)
    chem_desc[149] = Descriptors.fr_imide(m)
    chem_desc[150] = Descriptors.fr_isocyan(m)
    chem_desc[151] = Descriptors.fr_isothiocyan(m)
    chem_desc[152] = Descriptors.fr_ketone(m)
    chem_desc[153] = Descriptors.fr_ketone_Topliss(m)
    chem_desc[154] = Descriptors.fr_lactam(m)
    chem_desc[155] = Descriptors.fr_lactone(m)
    chem_desc[156] = Descriptors.fr_methoxy(m)
    chem_desc[157] = Descriptors.fr_morpholine(m)
    chem_desc[158] = Descriptors.fr_nitrile(m)
    chem_desc[159] = Descriptors.fr_nitro(m)
    chem_desc[160] = Descriptors.fr_nitro_arom(m)
    chem_desc[161] = Descriptors.fr_nitro_arom_nonortho(m)
    chem_desc[162] = Descriptors.fr_nitroso(m)
    chem_desc[163] = Descriptors.fr_oxazole(m)
    chem_desc[164] = Descriptors.fr_oxime(m)
    chem_desc[165] = Descriptors.fr_para_hydroxylation(m)
    chem_desc[166] = Descriptors.fr_phenol(m)
    chem_desc[167] = Descriptors.fr_phenol_noOrthoHbond(m)
    chem_desc[168] = Descriptors.fr_phos_acid(m)
    chem_desc[169] = Descriptors.fr_phos_ester(m)
    chem_desc[170] = Descriptors.fr_piperdine(m)
    chem_desc[171] = Descriptors.fr_piperzine(m)
    chem_desc[172] = Descriptors.fr_priamide(m)
    chem_desc[173] = Descriptors.fr_prisulfonamd(m)
    chem_desc[174] = Descriptors.fr_pyridine(m)
    chem_desc[175] = Descriptors.fr_quatN(m)
    chem_desc[176] = Descriptors.fr_sulfide(m)
    chem_desc[177] = Descriptors.fr_sulfonamd(m)
    chem_desc[178] = Descriptors.fr_sulfone(m)
    chem_desc[179] = Descriptors.fr_term_acetylene(m)
    chem_desc[180] = Descriptors.fr_tetrazole(m)
    chem_desc[181] = Descriptors.fr_thiazole(m)
    chem_desc[182] = Descriptors.fr_thiocyan(m)
    chem_desc[183] = Descriptors.fr_thiophene(m)
    chem_desc[184] = Descriptors.fr_unbrch_alkane(m)
    chem_desc[185] = Descriptors.fr_urea(m)
    chem_desc[186] = Descriptors.MolWt(m)  # 16
    chem_desc[187] = Descriptors.HeavyAtomMolWt(m)  # 18
    chem_desc[188] = Descriptors.NumValenceElectrons(m)  # 25
    chem_desc[189] = Descriptors.NumSaturatedRings(m)
    chem_desc = np.round(chem_desc, 4)
    chem_desc = chem_desc.tolist()
    
    return chem_desc
    
def get_reaction_site_smiles(metabolites):
    mol_folder = 'D:/Sahar/Work/Enzymes/Turnover/kcat_prediction/data/metabolite_data/mol-files'
    reaction_site = ""
    for met in metabolites:
        is_kegg_id = False

        if met[0] == "C":
            is_kegg_id = True

        if is_kegg_id:
            try:
                Smarts = Chem.MolToSmiles(Chem.MolFromMolFile(join(mol_folder, met + '.mol')))
            except OSError:
                return(np.nan)
        else:
            mol = Chem.inchi.MolFromInchi(met)
            if mol is not None:
                Smarts = Chem.MolToSmiles(mol)
            else:
                return(np.nan)
        reaction_site = reaction_site + "." + Smarts
    return(reaction_site[1:])    
    
data = pd.read_pickle(fpath+'merged_and_grouped_kcat_dataset.pkl')

for ind in data.index:
    
    substrates = list(data["substrates"][ind])
    products = list(data["products"][ind])
    seq = re.sub("X", "", re.sub("U", "", data.Sequence[ind]))
    AAcomp_dect, DPcomp_dect, MBA_dect, MA_dect, GA_dect, CTD_dect, PAAC_dect, \
    APAAC_dect, SOCN_dect, QSO_dect, traid_dect = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    try:
        AAcomp_dect.update(calc_AAcomp(seq))
    except Exception as e:
        print('AAcomp')
        print(seq)
        print(e)
        pass
    try:
        DPcomp_dect.update(calc_DPcomp(seq))
    except Exception as e:
        print('DPcomp')
        print(seq)
        print(e)
        pass
    try:
        MBA_dect.update(calc_MBA(seq))
    except Exception as e:
        print('MBA')
        print(seq)
        print(e)
        pass
    try:
        MA_dect.update(calc_MA(seq))
    except Exception as e:
        print('MA')
        print(seq)
        print(e)
        pass
    try:
        GA_dect.update(calc_GA(seq))
    except Exception as e:
        print('GA')
        print(seq)
        print(e)
        pass
    try:
        CTD_dect.update(calc_CTD(seq))
    except Exception as e:
        print('CTD')
        print(seq)
        print(e)
        pass
    try:
        PAAC_dect.update(calc_PAAC(seq))
    except Exception as e:
        print('PAAC')
        print(seq)
        print(e)
        pass
    try:
        APAAC_dect.update(calc_APAAC(seq))
    except Exception as e:
        print('APAAC')
        print(seq)
        print(e)
        pass
    try:
        SOCN_dect.update(calc_SOCN(seq))
    except Exception as e:
        print('SOCN')
        print(seq)
        print(e)
        pass
    try:
        QSO_dect.update(calc_QSO(seq))
    except Exception as e:
        print('QSO')
        print(seq)
        print(e)
        pass
    try:
        traid_dect.update(calc_traid(seq))
    except Exception as e:
        print('traid')
        print(seq)
        print(e)
        pass
    AAcomp = pd.concat([AAcomp, pd.DataFrame([AAcomp_dect])], ignore_index = True)
    DPcomp = pd.concat([DPcomp, pd.DataFrame([DPcomp_dect])], ignore_index = True)
    MBA = pd.concat([MBA, pd.DataFrame([MBA_dect])], ignore_index = True)
    MA = pd.concat([MA, pd.DataFrame([MA_dect])], ignore_index = True)
    GA = pd.concat([GA, pd.DataFrame([GA_dect])], ignore_index = True)
    CTD = pd.concat([CTD, pd.DataFrame([CTD_dect])], ignore_index = True)
    PAAC = pd.concat([PAAC, pd.DataFrame([PAAC_dect])], ignore_index = True)
    APAAC = pd.concat([APAAC, pd.DataFrame([APAAC_dect])], ignore_index = True)
    SOCN = pd.concat([SOCN, pd.DataFrame([SOCN_dect])], ignore_index = True)
    QSO = pd.concat([QSO, pd.DataFrame([QSO_dect])], ignore_index = True)
    traid = pd.concat([traid, pd.DataFrame([traid_dect])], ignore_index = True)
    
    
    left_maccs, left_chem =  np.zeros(167, dtype='float64'), np.zeros(190, dtype='float64')
    right_maccs, right_chem =  np.zeros(167, dtype='float64'), np.zeros(190, dtype='float64')
    maccs, chems = np.zeros(167, dtype='float64'), np.zeros(190, dtype='float64')
    try:
      left_site = get_reaction_site_smiles(substrates)
      right_site = get_reaction_site_smiles(products)
      if not pd.isnull(left_site) and not pd.isnull(right_site):
      
        left_smis = left_site.split('.')
        right_smis = right_site.split('.')
        for smile in left_smis:
            try:
                left_maccs = np.add(left_maccs, np.array(calc_maccs(smile), dtype='float64'))
            except Exception as e:
                pass
            try:
                left_chem = np.round(np.add(left_chem, np.array(calc_chems(smile), dtype='float64')), 4)
            except Exception as e:
                pass
        for smile in right_smis:
            try:
                right_maccs = np.add(right_maccs, np.array(calc_maccs(smile), dtype='float64'))
            except Exception as e:
                pass
            try:
                right_chem = np.round(np.add(right_chem, np.array(calc_chems(smile), dtype='float64')), 4)
            except Exception as e:
                pass
        try:
            maccs = np.subtract(left_maccs, right_maccs)
            chems = np.subtract(left_chem, right_chem)
        except:
            pass

    except IndexError:
      print(ind)

    maccs_keys_dect = {'maccs_key' + str(i): maccs[i] for i in range(len(maccs))}
    chemical_descriptors_dect = {'chem_des' + str(i): chems[i] for i in range(len(chems))}
    macs_keys = pd.concat([macs_keys, pd.DataFrame([maccs_keys_dect])], ignore_index = True)
    chemical_descriptors = pd.concat([chemical_descriptors, pd.DataFrame([chemical_descriptors_dect])], ignore_index = True)

print(data.shape[0], macs_keys.shape[0])
df = data.join(macs_keys)
df = df.join(chemical_descriptors)
df = df.join(AAcomp)
df = df.join(DPcomp) 
df = df.join(MBA)
df = df.join(MA)
df = df.join(GA)
df = df.join(CTD)
df = df.join(PAAC)
df = df.join(APAAC, lsuffix='_PAAC', rsuffix= '_APAAC')
df = df.join(SOCN)
df = df.join(QSO)
df = df.join(traid)

df.to_csv(fpath+'new_features.csv', index=False)
df.to_pickle(fpath+'new_features.pkl')


