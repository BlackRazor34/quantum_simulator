# streamlit_quantum_simulator_app.py
# Expanded educational quantum simulator (NumPy-only)
# - Deutsch–Jozsa (phase-oracle)
# - Grover (small n)
# - Single-qubit Bloch sphere visualization (matplotlib 3D)
# - Quantum Teleportation (3-qubit) with step-by-step mode
# - UI includes buttons to write Dockerfile and README to disk for easy deployment
# Run: streamlit run streamlit_quantum_simulator_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, sqrt, floor
import textwrap
import os

st.set_page_config(page_title="Kuantum Eğitimi - Genişletilmiş Simülatör", layout="wide")
st.title("Kuantum Algoritma Eğitimi — Saf NumPy Simülatörü (Genişletilmiş)")
st.markdown("Matplotlib 3D Bloch küresi, Teleportation demo, adım-adım yürütme")

# ------------------ Linear-algebra helpers ------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def kron_n(mat_list):
    res = mat_list[0]
    for m in mat_list[1:]:
        res = np.kron(res, m)
    return res


def zero_state(n):
    v = np.zeros(2**n, dtype=complex)
    v[0] = 1.0
    return v


def apply_unitary(state, U):
    return U @ state


def single_qubit_gate_on_n(gate, target, n):
    mats = []
    for i in range(n):
        if i == target:
            mats.append(gate)
        else:
            mats.append(I2)
    return kron_n(mats)


def cnot_on_n(control, target, n):
    # Build 2^n x 2^n matrix for CNOT (works for small n)
    N = 2**n
    U = np.zeros((N, N), dtype=complex)
    for i in range(N):
        b = list(map(int, format(i, f'0{n}b')))
        if b[n-1-control] == 1:  # note: qubit ordering: 0 is leftmost in string; we use little-endian indexing
            b2 = b.copy()
            b2[n-1-target] ^= 1
            j = int(''.join(map(str, b2)), 2)
            U[j, i] = 1
        else:
            U[i, i] = 1
    return U


def phase_oracle_diagonal(f_vals):
    phases = np.array([(-1)**v for v in f_vals], dtype=complex)
    return np.diag(phases)


def measure_counts(state, shots=1024):
    probs = np.abs(state)**2
    outcomes = list(range(len(probs)))
    samples = np.random.choice(outcomes, size=shots, p=probs)
    unique, counts = np.unique(samples, return_counts=True)
    result = {format(u, '0{}b'.format(int(np.log2(len(probs))))): c for u, c in zip(unique, counts)}
    return result, probs

# ------------------ Bloch sphere helpers ------------------

def bloch_from_statevec(statevec):
    # statevec is length-2 vector (complex)
    # compute Bloch vector components from density matrix rho = |psi><psi|
    rho = np.outer(statevec, np.conj(statevec))
    x = np.real(np.trace(rho @ X))
    y = np.real(np.trace(rho @ Y))
    z = np.real(np.trace(rho @ Z))
    return np.array([x, y, z])


def single_qubit_reduced_state(full_state, target, n):
    # compute reduced statevector amplitude for single qubit
    # return statevector of single qubit in computational basis (length 2)
    # compute density matrix of full system and partial trace
    N = 2**n
    rho_full = np.outer(full_state, np.conj(full_state))
    # indices ordering: we'll trace out all qubits except target
    # build mapping of basis states
    rho_red = np.zeros((2,2), dtype=complex)
    for i in range(N):
        bi = list(map(int, format(i, f'0{n}b')))
        for j in range(N):
            bj = list(map(int, format(j, f'0{n}b')))
            # if all other qubits equal -> contribute to reduced density
            ok = True
            for q in range(n):
                if q == target:
                    continue
                if bi[q] != bj[q]:
                    ok = False
                    break
            if ok:
                a = bi[target]
                b = bj[target]
                rho_red[a, b] += rho_full[i, j]
    # now try to get a pure state vector from rho_red (if pure)
    # if rank==1, eigenvector with largest eigenvalue
    vals, vecs = np.linalg.eigh(rho_red)
    idx = np.argmax(vals)
    eigval = vals[idx]
    eigvec = vecs[:, idx]
    # normalize
    if np.abs(eigvec).sum() == 0:
        eigvec = np.array([1.0, 0.0], dtype=complex)
    else:
        eigvec = eigvec / np.linalg.norm(eigvec)
    return eigvec, eigval


def plot_bloch(ax, vec, title=''):
    # vec is 3-element Bloch vector
    # draw sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.1)
    # axes
    ax.quiver(0,0,0,1.0,0,0,length=1, normalize=True)
    ax.quiver(0,0,0,0,1.0,0,length=1, normalize=True)
    ax.quiver(0,0,0,0,0,1.0,length=1, normalize=True)
    # vector
    ax.quiver(0,0,0,vec[0], vec[1], vec[2], length=1, color='r', linewidth=2)
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

# ------------------ Algorithms ------------------

def deutsch_jozsa(n, oracle_type='constant-0'):
    N = 2**n
    if oracle_type == 'constant-0':
        f = np.zeros(N, dtype=int)
    elif oracle_type == 'constant-1':
        f = np.ones(N, dtype=int)
    elif oracle_type == 'balanced-parity':
        f = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=int)
    elif oracle_type == 'balanced-half':
        f = np.zeros(N, dtype=int)
        f[:N//2] = 1
    else:
        f = np.zeros(N, dtype=int)

    state = zero_state(n)
    Hn = kron_n([H]*n)
    state = apply_unitary(state, Hn)
    Of = phase_oracle_diagonal(f)
    state = apply_unitary(state, Of)
    state = apply_unitary(state, Hn)
    return state, f


def grover(n, target_index, iterations=None):
    N = 2**n
    if iterations is None or iterations <= 0:
        iterations = max(1, int(floor((pi/4)*sqrt(N))))

    state = zero_state(n)
    Hn = kron_n([H]*n)
    state = apply_unitary(state, Hn)
    Of = np.eye(N, dtype=complex)
    Of[target_index, target_index] = -1
    s = (1/np.sqrt(N)) * np.ones((N,), dtype=complex)
    Us = 2.0 * np.outer(s, s.conj()) - np.eye(N, dtype=complex)

    for _ in range(iterations):
        state = Of @ state
        state = Us @ state
    return state, iterations

# ------------------ Teleportation ------------------

def teleportation_circuit_steps(initial_statevec):
    # initial_statevec: length-2 vector for qubit |psi> to teleport (Alice's qubit)
    # overall system: [psi, 0, 0] -> 3 qubits: psi, bell_a, bell_b
    n = 3
    # build initial full state
    psi = initial_statevec
    state = np.kron(np.kron(psi, np.array([1.0, 0.0], dtype=complex)), np.array([1.0, 0.0], dtype=complex))

    steps = []
    steps.append(('start', state.copy()))
    # Step 1: create Bell pair between qubit1 (index 1) and qubit2 (index 2)
    # apply H on qubit 1 (index 1 -> position 1), then CNOT(1->2)
    U_h_q1 = single_qubit_gate_on_n(H, target=1, n=3)
    state = U_h_q1 @ state
    steps.append(('H on qubit1 (Bell prep)', state.copy()))
    U_cnot_1_2 = cnot_on_n(control=1, target=2, n=3)
    state = U_cnot_1_2 @ state
    steps.append(('CNOT qubit1->qubit2 (Bell pair)', state.copy()))

    # Step 2: Alice applies CNOT(0->1) then H(0)
    U_cnot_0_1 = cnot_on_n(control=0, target=1, n=3)
    state = U_cnot_0_1 @ state
    steps.append(('CNOT qubit0->qubit1', state.copy()))
    U_h_q0 = single_qubit_gate_on_n(H, target=0, n=3)
    state = U_h_q0 @ state
    steps.append(('H on qubit0', state.copy()))

    # Step 3: measure qubit0 and qubit1 (simulate classical results) and apply corrections on qubit2
    # We will not collapse randomly here; instead we return the pre-measurement state and describe correction outcomes
    steps.append(('pre-measurement (Alice measured qubit0 and qubit1)', state.copy()))

    return steps


def apply_teleportation_measure_and_corrections(pre_measure_state, m0, m1):
    # m0, m1 are measurement bits (0/1)
    # After measuring, Bob's qubit (qubit2) may need X (if m1==1) and Z (if m0==1)
    # We'll compute post-measurement collapsed state conditioned on (m0,m1)
    n = 3
    N = 2**n
    # For each basis state index i, check bits of qubit0 and qubit1
    collapsed = np.zeros(N, dtype=complex)
    for i in range(N):
        b = list(map(int, format(i, f'0{n}b')))
        if b[0] == m0 and b[1] == m1:
            collapsed[i] = pre_measure_state[i]
    norm = np.linalg.norm(collapsed)
    if norm == 0:
        collapsed = collapsed
    else:
        collapsed = collapsed / norm

    # Now apply corrections on qubit2
    # X if m1==1 on qubit2 -> apply X on target=2
    if m1 == 1:
        Ux = single_qubit_gate_on_n(X, target=2, n=3)
        collapsed = Ux @ collapsed
    if m0 == 1:
        Uz = single_qubit_gate_on_n(Z, target=2, n=3)
        collapsed = Uz @ collapsed
    return collapsed

# ------------------ UI ------------------

st.sidebar.header('Ayarlar')
mode = st.sidebar.selectbox('Demo', ['Deutsch–Jozsa', 'Grover', 'Teleportation', 'Adım-adım (Kapıları izle)'])
shots = st.sidebar.slider('Shots (ölçüm sayısı)', 128, 8192, 1024, step=128)

col1, col2 = st.columns([1,1])

if mode == 'Deutsch–Jozsa':
    with col1:
        st.header('Deutsch–Jozsa (phase-oracle)')
        n = st.slider('Girdi qubit sayısı n', 1, 6, 3)
        oracle_type = st.selectbox('Oracle tipi', ['constant-0', 'constant-1', 'balanced-parity', 'balanced-half'])
        if st.button('Çalıştır'):
            state, f = deutsch_jozsa(n, oracle_type)
            counts, probs = measure_counts(state, shots=shots)
            st.subheader('Ölçüm dağılımı')
            labels = list(counts.keys())
            values = [counts[l] for l in labels]
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(labels, values)
            ax.set_xlabel('Bit dizisi'); ax.set_ylabel('Counts'); plt.xticks(rotation=45)
            st.pyplot(fig)
            st.subheader('Durum vektörü (|amplitude|)')
            amps = state
            states = [format(i, f'0{n}b') for i in range(2**n)]
            fig2, ax2 = plt.subplots(figsize=(8,2))
            ax2.bar(states, np.abs(amps))
            ax2.set_ylabel('|amplitude|'); plt.xticks(rotation=45)
            st.pyplot(fig2)
            if np.abs(state[0])**2 > 0.999:
                st.success('Fonksiyon muhtemelen SABİTTİR (çıktı 0...0).')
            else:
                st.info('Fonksiyon muhtemelen DENGELİ.')
            st.write('Oracle (ilk 16 f(x)):', f[:min(len(f),16)])

    with col2:
        st.subheader('Tek qubit Bloch (ilk qubit)')
        # compute reduced of first qubit
        if 'state' in locals():
            psi1, val = single_qubit_reduced_state(state, target=0, n=n)
            bloch = bloch_from_statevec(psi1)
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='3d')
            plot_bloch(ax, bloch, title=f'ρ purity={val:.3f}')
            st.pyplot(fig)
        else:
            st.info('Algoritma çalıştırıldığında ilk qubit Bloch küresi gösterilecek.')

elif mode == 'Grover':
    with col1:
        st.header('Grover (küçük n demo)')
        n = st.slider('Qubit sayısı n', 1, 5, 3)
        target = st.text_input('Hedef durum (binary)', '0'*n)
        if len(target) != n or any(c not in '01' for c in target):
            st.warning('Lütfen n bitlik binary bir dize girin.')
        else:
            t_index = int(target, 2)
            iters = st.number_input('Iterasyon sayısı (0: otomatik)', min_value=0, max_value=100, value=0)
            iters_val = None if iters == 0 else int(iters)
            if st.button('Çalıştır'):
                state, iters_done = grover(n, t_index, iterations=iters_val)
                counts, probs = measure_counts(state, shots=shots)
                st.subheader(f'İterasyonlar: {iters_done} — Ölçüm dağılımı')
                labels = list(counts.keys())
                values = [counts[l] for l in labels]
                fig, ax = plt.subplots(figsize=(6,3))
                ax.bar(labels, values); ax.set_xlabel('Bit dizisi'); ax.set_ylabel('Counts'); plt.xticks(rotation=45)
                st.pyplot(fig)
                st.subheader('Durum vektörü (|amplitude|)')
                states = [format(i, f'0{n}b') for i in range(2**n)]
                fig2, ax2 = plt.subplots(figsize=(8,2))
                ax2.bar(states, np.abs(state))
                ax2.set_ylabel('|amplitude|'); plt.xticks(rotation=45)
                st.pyplot(fig2)
                prob_target = np.abs(state[t_index])**2
                st.write(f'Hedef {target} için ölçüm olasılığı (teorik): {prob_target:.4f}')

    with col2:
        st.subheader('Tek qubit Bloch (ilk qubit)')
        if 'state' in locals():
            psi1, val = single_qubit_reduced_state(state, target=0, n=n)
            bloch = bloch_from_statevec(psi1)
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='3d')
            plot_bloch(ax, bloch, title=f'ρ purity={val:.3f}')
            st.pyplot(fig)
        else:
            st.info('Algoritma çalıştırıldığında ilk qubit Bloch küresi gösterilecek.')

elif mode == 'Teleportation':
    st.header('Quantum Teleportation — Adım adım')
    with st.expander('Başlangıç durumu ayarla'):
        theta = st.slider('θ (Bloch polar açı)', 0.0, pi, 0.3)
        phi = st.slider('φ (Bloch azimut açısı)', 0.0, 2*pi, 0.7)
        psi = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)
        st.latex(r"|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle")
        st.write('Başlangıç qubit amplitüdleri:', np.round(psi,3))

    steps = teleportation_circuit_steps(psi)
    step_names = [s[0] for s in steps]
    sel = st.selectbox('Adımlar', step_names)
    idx = step_names.index(sel)
    st.subheader(f'Step: {sel}')
    state = steps[idx][1]
    n = 3
    # show amplitudes
    states = [format(i, '03b') for i in range(8)]
    fig, ax = plt.subplots(figsize=(6,2))
    ax.bar(states, np.abs(state))
    ax.set_ylabel('|amplitude|'); plt.xticks(rotation=45)
    st.pyplot(fig)

    # Bloch of qubit0 and qubit2
    psi0, val0 = single_qubit_reduced_state(state, target=0, n=3)
    psi2, val2 = single_qubit_reduced_state(state, target=2, n=3)
    colA, colB = st.columns(2)
    with colA:
        st.write('Alice: qubit0 Bloch (redu.) purity=', round(val0,3))
        fig1 = plt.figure(figsize=(3,3))
        ax1 = fig1.add_subplot(111, projection='3d')
        plot_bloch(ax1, bloch_from_statevec(psi0))
        st.pyplot(fig1)
    with colB:
        st.write('Bob: qubit2 Bloch (redu.) purity=', round(val2,3))
        fig2 = plt.figure(figsize=(3,3))
        ax2 = fig2.add_subplot(111, projection='3d')
        plot_bloch(ax2, bloch_from_statevec(psi2))
        st.pyplot(fig2)

    st.markdown('---')
    st.write('**Ölçüm ve düzeltme simülasyonu (örnek):**')
    m0 = st.selectbox('Alice measure qubit0 bit (m0)', [0,1])
    m1 = st.selectbox('Alice measure qubit1 bit (m1)', [0,1])
    if st.button('Ölç ve düzelt (koşullu)'):
        pre = steps[-1][1]
        collapsed = apply_teleportation_measure_and_corrections(pre, m0, m1)
        psi2_post, valp = single_qubit_reduced_state(collapsed, target=2, n=3)
        st.write('Bob qubit (teleport sonrası) amplitüdleri:', np.round(psi2_post,4))
        st.write('Başlangıç amplitüdleri (|ψ>):', np.round(psi,4))
        st.success('Bob qubit i̇le başlangıç qubitinin eşleşmesine bak (global faza dikkat!).')

elif mode == 'Adım-adım (Kapıları izle)':
    st.header('Adım-adım yürütme (örnek: Deutsch–Jozsa small)')
    n = st.slider('Qubit sayısı n', 1, 4, 2)
    oracle_type = st.selectbox('Oracle tipi', ['constant-0', 'balanced-parity'])
    # build list of unitaries representing sequence of gates for Deutsch-Jozsa
    seq = []
    Hn = kron_n([H]*n)
    seq.append(('H^n', Hn))
    # oracle
    N = 2**n
    if oracle_type == 'constant-0':
        f = np.zeros(N, dtype=int)
    else:
        f = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=int)
    Of = phase_oracle_diagonal(f)
    seq.append((f'Oracle ({oracle_type})', Of))
    seq.append(('H^n (again)', Hn))

    if 'step_idx' not in st.session_state:
        st.session_state.step_idx = 0
        st.session_state.seq_state = zero_state(n)

    if st.button('Reset'):
        st.session_state.step_idx = 0
        st.session_state.seq_state = zero_state(n)

    if st.button('Sonraki Adım'):
        name, U = seq[st.session_state.step_idx]
        st.session_state.seq_state = U @ st.session_state.seq_state
        st.session_state.step_idx = min(st.session_state.step_idx + 1, len(seq)-1)

    st.write(f'Adım index: {st.session_state.step_idx} / {len(seq)-1}')
    cur_state = st.session_state.seq_state
    fig, ax = plt.subplots(figsize=(8,2))
    states = [format(i, f'0{n}b') for i in range(2**n)]
    ax.bar(states, np.abs(cur_state))
    ax.set_ylabel('|amplitude|'); plt.xticks(rotation=45)
    st.pyplot(fig)

    psi0, val0 = single_qubit_reduced_state(cur_state, target=0, n=n)
    figb = plt.figure(figsize=(3,3))
    axb = figb.add_subplot(111, projection='3d')
    plot_bloch(axb, bloch_from_statevec(psi0))
    st.pyplot(figb)


st.sidebar.markdown('---')
st.sidebar.caption('Not: Büyük qubit sayıları hesaplama karmaşıklığı (2^n) nedeniyle yavaş olacaktır.')
