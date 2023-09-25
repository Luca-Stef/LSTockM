from train_utils import *

symbol = sys.argv[1]

# shortable_etfs = ['aaxj', 'acwi', 'acwv', 'acwx', 'agg', 'agq', 'aia', 'amj', 'amlp', 'angl', 'aoa', 'aom', 'aor', 'ashr', 'bab', 'bbh', 'bil', 'biv', 'bkln', 'blv', 'bnd', 'bndx', 'boil', 'bsv', 'bwx', 'bwz', 'chiq', 'comt', 'cqqq', 'cwb', 'cwi', 'dba', 'dbc', 'dbp', 'dem', 'des', 'dfj', 'dgro', 'dgrw', 'dgs', 'dia', 'div', 'djp', 'drip', 'dsi', 'dvy', 'dvye', 'dxj', 'ebnd', 'ech', 'ecns', 'eem', 'eema', 'eems', 'eemv', 'efa', 'efav', 'efg', 'efv', 'eido', 'eis', 'emb', 'emlc', 'enzl', 'ephe', 'epi', 'epol', 'epu', 'eqal', 'ero', 'eufn', 'ewa', 'ewc', 'ewg', 'ewh', 'ewj', 'ewl', 'ewm', 'ewo', 'ewp', 'ewq', 'ews', 'ewt', 'ewu', 'eww', 'ewx', 'ewy', 'ewz', 'eza', 'ezu', 'fbt', 'fcg', 'fdl', 'fdn', 'fez', 'fhlc', 'flot', 'flrn', 'fltr', 'fm', 'fncl', 'fndc', 'fnde', 'fndf', 'fnx', 'fpe', 'fpx', 'fta', 'ftc', 'ftcs', 'ftgc', 'ftsm', 'futy', 'fvd', 'fxg', 'fxh', 'fxi', 'fxl', 'fxn', 'fxr', 'fxu', 'fxz', 'fyx', 'gdx', 'gdxj', 'gii', 'gld', 'gnr', 'govt', 'grek', 'gsg', 'gunr', 'gvi', 'gwx', 'gxg', 'hdv', 'hedj', 'hyem', 'hyg', 'iai', 'iak', 'iat', 'iau', 'ibb', 'ibdo', 'ibdp', 'icf', 'icln', 'idv', 'ief', 'iefa', 'iei', 'iemg', 'ieo', 'ieur', 'iez', 'ige', 'igf', 'igm', 'ign', 'igv', 'ihdg', 'ihf', 'ihi', 'ijh', 'ijj', 'ijk', 'ijr', 'ijs', 'ijt', 'ilf', 'iltb', 'imtm', 'inda', 'indy', 'ioo', 'ipac', 'ipo', 'iqlt', 'iscf', 'istb', 'ita', 'itb', 'itot', 'iusb', 'iusg', 'iusv', 'ive', 'ivoo', 'ivv', 'ivw', 'iwb', 'iwc', 'iwd', 'iwf', 'iwl', 'iwm', 'iwn', 'iwo', 'iwp', 'iwr', 'iws', 'iwv', 'iwy', 'ixc', 'ixg', 'ixj', 'ixn', 'ixus', 'iyc', 'iye', 'iyf', 'iyg', 'iyh', 'iyj', 'iyk', 'iym', 'iyr', 'iyt', 'iyw', 'iyy', 'iyz', 'jnk', 'kba', 'kbe', 'kce', 'kie', 'kre', 'labu', 'lemb', 'lit', 'lmbs', 'lqd', 'mbb', 'mchi', 'mdyv', 'mgc', 'mint', 'mna', 'moat', 'moo', 'mort', 'mtum', 'near', 'nfra', 'nobl', 'oef', 'oih', 'olo', 'oneq', 'ounz', 'pbj', 'pbw', 'pcy', 'pdbc', 'pej', 'pey', 'pff', 'pfm', 'pfxf', 'pgx', 'pick', 'pjp', 'pkb', 'pnqi', 'ppa', 'pph', 'prfz', 'pscc', 'psct', 'psk', 'pwb', 'pwv', 'pxe', 'pxh', 'qat', 'qid', 'qld', 'qlta', 'qqew', 'qqq', 'qqqe', 'qtec', 'qual', 'qus', 'qyld', 'rdvy', 'reet', 'ring', 'rpg', 'rpv', 'rsp', 'rwl', 'rwo', 'rwr', 'sbio', 'scha', 'schb', 'schc', 'schd', 'sche', 'schf', 'schg', 'schh', 'schm', 'scho', 'schp', 'schr', 'schv', 'schx', 'schz', 'sco', 'scz', 'sdiv', 'sdog', 'sgol', 'shv', 'shy', 'shyd', 'shyg', 'sil', 'silj', 'sivr', 'size', 'sjnk', 'skyy', 'slqd', 'slv', 'slyg', 'slyv', 'smdv', 'smh', 'smlv', 'snln', 'soxl', 'soxs', 'soxx', 'spff', 'sphb', 'sphd', 'sphq', 'splv', 'spuu', 'spxu', 'spy', 'spyg', 'spyv', 'sqqq', 'srln', 'sso', 'stip', 'sub', 'svxy', 'tan', 'tbf', 'tflo', 'thd', 'tip', 'tipx', 'tlh', 'tlt', 'tmf', 'totl', 'tqqq', 'tza', 'uae', 'uco', 'ung', 'upro', 'ura', 'urth', 'usfr', 'usmv', 'uup', 'uvxy', 'vaw', 'vb', 'vbk', 'vbr', 'vcit', 'vclt', 'vcr', 'vcsh', 'vdc', 'vde', 'vea', 'veu', 'vfh', 'vgit', 'vgk', 'vglt', 'vgsh', 'vgt', 'vht', 'vig', 'vioo', 'viov', 'vlu', 'vlue', 'vmbs', 'vnm', 'vnq', 'vnqi', 'vo', 'voe', 'vone', 'vong', 'vonv', 'voo', 'voog', 'voov', 'vot', 'vox', 'vpl', 'vpu', 'vss', 'vt', 'vthr', 'vti', 'vtip', 'vtv', 'vtwo', 'vtwv', 'vug', 'vv', 'vwo', 'vwob', 'vxf', 'vxus', 'vxx', 'vym', 'wip', 'xar', 'xbi', 'xes', 'xhb', 'xhe', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xlv', 'xly', 'xme', 'xmlv', 'xop', 'xph', 'xrt', 'xsd', 'xslv', 'xsoe', 'xsw', 'xt', 'xtl', 'xtn', 'yinn']

# for symbol in shortable_etfs[184:]:
    
X_raw, y_raw, scaler = get_batch(symbol)

if isinstance(X_raw, np.ndarray):

    X, y = create_windows(X_raw, y_raw)
    
    if len(X.shape) < 3:

        X = X.reshape(*X.shape, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    breakpoint()
    model = LSTM_model(X_train, y_train)
    loss, accuracy = fit_evaluate_LSTM(X_train, y_train, X_test, y_test, model, symbol, epochs=200)
    model.save(f'models/{symbol}_{accuracy}')
    dump(scaler, open(f'models/{symbol}_scaler.pkl', 'wb'))