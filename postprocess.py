import pandas as pd
from numpy import pi, log2
import hashlib

theta = 0.0
mesh_size = [1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
phys_pars = {'E': 1, 'nu_s': 0.3, 'pre': 4e-2, 'pre_0': 0, 'alpha_B': 0,
             'theta_0': 0.0, 'theta': theta, 'alpha_th': 1e-5}

hash_pars = hashlib.blake2b(str(phys_pars).encode('utf-8'), digest_size=6).hexdigest()


mu_s = phys_pars['E'] / (2 * (1 + phys_pars['nu_s']))
lam_s = phys_pars['nu_s'] * phys_pars['E'] / ((1 + phys_pars['nu_s']) * (1 - 2 * phys_pars['nu_s']))
k_dr = 2 / 3 * mu_s + lam_s

filename = f'results/example1_thermal_phase_field_fracture_h{mesh_size[-1]}_{hash_pars}'
df_cod = pd.read_csv(f'{filename}_cod.dat', sep=' ')
cod_0 = df_cod.loc[df_cod.x == 0.0, 'cod'].values[0]
tcv_0 = pd.read_csv(f'{filename}_tcv.dat', header=None).iloc[0].values[0]

err_cod = []
err_tcv = []
rate_cod, rate_tcv = 0, 0

file_out = f'results/example1_thermal_phase_field_fracture_{hash_pars}_convergence.dat'
fid = open(file_out, 'w')
fid.write('lvl h cod coderr codrate tcv tcverr tcvrate\n')

for i, h in enumerate(mesh_size[:-1]):
    filename = f'results/example1_thermal_phase_field_fracture_h{h}_{hash_pars}'

    df_cod = pd.read_csv(f'{filename}_cod.dat', sep=' ')
    cod = df_cod.loc[df_cod.x == 0.0, 'cod'].values[0]
    err_cod.append(abs(cod_0 - cod))

    tcv = pd.read_csv(f'{filename}_tcv.dat', header=None).iloc[0].values[0]
    err_tcv.append(abs(tcv_0 - tcv))

    if i > 0:
        rate_cod = log2(err_cod[i - 1]) - log2(err_cod[i])
        rate_tcv = log2(err_tcv[i - 1]) - log2(err_tcv[i])

    fid.write(f'{i} {h} {cod:.6e} {err_cod[-1]:.3e} {rate_cod:4.2f} ')
    fid.write(f'{tcv:.6e} {err_tcv[-1]:.3e} {rate_tcv:4.2f}\n')

fid.close()
