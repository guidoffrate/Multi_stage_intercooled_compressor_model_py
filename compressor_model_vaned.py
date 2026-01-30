import math
import numpy as np
from CoolProp import AbstractState, PT_INPUTS, DmassT_INPUTS, HmassP_INPUTS, HmassSmass_INPUTS
from scipy.optimize import root
from scipy.optimize import brentq
import pandas as pd


class CentrifugalCompressor:
    """
    A Python class for modeling a centrifugal compressor with vaned diffuser.
    
    The compressor sections are:
        - 1 -> Impeller inlet
        - 2 -> Impeller outlet
        - 3 -> Vaned diffuser inlet
        - 4 -> Vaned diffuser outlet
        - 5 -> Volute outlet
        - 6 -> Cone outlet
    
    """

    def __init__(self):
        # Geometry and blade design
        self.N_blades_impeller = None
        self.N_blades_diffuser = None
        self.imp_blade_thickness_inlet = None
        self.imp_blade_thickness_outlet = None
        self.roughness = None
        self.disk_clearance = None
        self.radial_clearance = None
        self.axial_length = None

        # Local radius
        self.r1_shroud = None
        self.r1_hub = None
        self.r1_rms = None
        self.r2 = None
        self.r3 = None
        self.r4 = None
        self.r5 = None
        self.r6 = None

        # Heigth of the flow channel
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.b4 = None
        self.b5 = None
        self.b6 = None
        
        # Discharge cone geometry
        self.cone_diameter_outlet = None
        self.cone_diameter_inlet = None
        self.cone_length = None

        # Flow cross areas
        self.A1 = None
        self.A2 = None
        self.A3 = None
        self.A4 = None
        self.A5 = None
        self.A6 = None
        self.A7 = None

        # Mass flow and speed
        self.mdot = None
        self.omega_rpm = None
        self.omega_rads = None

        # Flow angles
        self.alpha1 = None
        self.beta1_blade_rms = None
        self.beta1_blade_shroud = None
        self.beta1_blade_hub = None
        self.beta1_rms = None
        self.beta1_shroud = None
        self.beta1_hub = None

        self.beta2_blade = None
        self.alpha2 = None

        self.alpha3 = None

        self.alpha4_blade = None
        self.alpha4 = None

        self.alpha5_blade = None
        self.alpha5 = None

        # Velocities
        self.u1_rms = None
        self.u1_hub = None
        self.u1_shroud = None
        self.c1 = None
        self.c1_meridional = None
        self.c1_tangential = None
        self.w1_rms = None
        self.w1_shroud = None
        self.w1_hub = None

        self.c1_throat = None
        self.w1_throat = None

        self.u2 = None
        self.c2 = None
        self.c2_meridional = None
        self.c2_tangential = None
        self.w2 = None

        self.c3 = None
        self.c3_meridional = None
        self.c3_tangential = None
        self.c3_is = None

        self.c4 = None
        self.c4_meridional = None

        self.c5 = None

        self.c6 = None

        self.c7 = None

        # Thermodynamic properties
        # Impeller inlet
        self.T_01 = None
        self.p_01 = None
        self.h_01 = None
        self.s_01 = None
        self.rho_01 = None
        self.T_1 = None
        self.p_1 = None
        self.h_1 = None
        self.s_1 = None
        self.rho_1 = None
        self.rho_1throat = None
        self.T_1throat = None
        self.p_1throat = None
        self.h_1throat = None
        self.Ma_1 = None

        # Impeller outlet
        self.T_2 = None
        self.p_2 = None
        self.h_2 = None
        self.s_2 = None
        self.rho_2 = None
        self.T_02 = None
        self.p_02 = None
        self.h_02 = None
        self.h_02is = None
        self.T_02is = None
        self.T_2is = None
        self.h_2is = None
        self.p_02is = None
        self.p_2 = None
        self.Ma_2 = None

        # Vaned diffuser inlet
        self.T_3 = None
        self.T_03 = None
        self.h_3 = None
        self.h_03 = None
        self.s_3 = None
        self.rho_3 = None
        self.Re_3 = None
        self.T_3is = None
        self.h_3is = None
        self.p_03 = None
        self.p_03is = None
        self.rho_3is = None

        # Vaned diffuser outlet
        self.T_4 = None
        self.T_04 = None
        self.h_4 = None
        self.h_04 = None
        self.s_4 = None
        self.rho_4 = None
        self.p_4 = None
        self.p_04 = None
        self.Ma_4 = None

        # Volute outlet
        self.T_5 = None
        self.T_05 = None
        self.h_5 = None
        self.h_05 = None
        self.s_5 = None
        self.rho_5 = None
        self.p_5 = None
        self.p_05 = None
        self.Ma_5 = None

        # Cone outlet
        self.T_6 = None
        self.T_06 = None
        self.h_6 = None
        self.h_06 = None
        self.s_6 = None
        self.rho_6 = None
        self.p_6 = None
        self.p_06 = None
        self.Ma_6 = None

        # Losses and performance
        self.L_eulero = None
        self.Dh_vaned_diffuser = None
        self.Dh_loss_volute = None
        self.Dh_internal_loss = None
        self.Dh_external_loss = None
        
        self.cf = None

        self.eta_is_tt = None
        self.eta_is_ts = None
        self.eta_pol_tt = None
        self.eta_pol_ts = None
        self.phi = None
        self.psi = None
        self.psi_eu = None
        self.PR_tt = None
        self.PR_ts = None
        self.PR_ss = None
        self.RR = None

        self.W = None
        self.Wdot = None

        # Fluid properties
        self.fluid = None
        self.fluid_as = None

        # Simulation settings
        self.convergence_1 = None
        self.convergence_2 = None
        self.convergence_3 = None
        self.convergence_4 = None
        self.convergence_5 = None
        self.convergence_6 = None
        self.verbose = False

    def set_geometry(self, geometry_specs):
        """
        set_geometry Sets the compressor geometrical specifications
        """

        self.N_blades_impeller = geometry_specs['N_blades_impeller']
        self.N_blades_diffuser = geometry_specs['N_blades_diffuser']
        # self.blade_thickness = geometry_specs['blade_thickness']
        self.imp_blade_thickness_inlet = geometry_specs['imp_blade_thickness_inlet']
        self.imp_blade_thickness_outlet = geometry_specs['imp_blade_thickness_outlet']

        self.diff_blade_thickness_inlet = geometry_specs['diff_blade_thickness_inlet']
        self.diff_blade_thickness_outlet = geometry_specs['diff_blade_thickness_outlet']

        self.disk_clearance = 0.00025  # (m) - imposed here. It can be moved outside.
        self.radial_clearance = 0.00025  # (m) - imposed here. It can be moved outside.

        self.roughness = geometry_specs['roughness']

        self.r1_shroud = geometry_specs['r1_shroud']
        self.r1_hub = geometry_specs['r1_hub']
        self.r1_rms = ((self.r1_shroud ** 2 + self.r1_hub ** 2) / 2) ** 0.5
        self.b1 = self.r1_shroud - self.r1_hub
        self.A1 = (
            math.pi * (self.r1_shroud ** 2 - self.r1_hub ** 2) -
            self.N_blades_impeller * self.imp_blade_thickness_outlet * (self.r1_shroud - self.r1_hub))
        self.alpha1 = geometry_specs['alpha1']
        self.beta1_blade_rms = geometry_specs['beta1_blade_rms']

        self.r2 = geometry_specs['r2']
        self.b2 = geometry_specs['b2']
        self.A2 = (2 * math.pi * self.r2 - self.N_blades_impeller * self.imp_blade_thickness_outlet) * self.b2
        self.beta2_blade = geometry_specs['beta2_blade']

        self.r3 = self.r2
        self.b3 = self.b2
        self.A3 = (2 * math.pi * self.r3 - self.N_blades_diffuser * self.diff_blade_thickness_inlet) * self.b3
        self.alpha3_blade = geometry_specs['alpha3_blade']

        self.r4 = geometry_specs['r4']
        self.b4 = self.b3
        self.A4 = (2 * math.pi * self.r4 - self.N_blades_diffuser * self.diff_blade_thickness_outlet) * self.b4
        self.alpha4_blade = geometry_specs['alpha4_blade']

        # Volute sizing
        # self.r6 = geometry_specs['r6']
        # self.r6 = self.r5 * 1.2
        # self.A6 = math.pi * (self.r6 - self.r5) ** 2
        
        # (Aungier, 2000) - doi:10.1115/1.800938 - page 125
        def f0_volute_sizing(r5_guess):
            SP = 3
            A5_1 = SP * self.A4 / self.r4 * math.tan(math.radians(self.alpha4_blade)) * r5_guess
            A5_2 = math.pi * (r5_guess - self.r4) ** 2
            res = A5_1 - A5_2
            return res
        self.r5 = brentq(f0_volute_sizing, self.r4, self.r4 * 4, xtol=1e-6)
        self.A5 = math.pi * (self.r5 - self.r4) ** 2

        #self.axial_length = 0.4 * (2 * self.r2 - (2 * self.r1_shroud + 2 * self.r1_hub) / 2)

    
    def set_fluid(self, fluid, backend=None):
        """
        Set the working fluid and (optionally) the backend for CoolProp AbstractState.
    
        Parameters
        ----------
        fluid : str
            Fluid string (e.g. "CO2").
        backend : str | None 
            - None -> use HEOS (default).
            - "HEOS", "REFPROP", "INCOMP", ... -> use that backend.
        """
        self.fluid = fluid
        
        # Default to HEOS if not specified/False
        if backend is None:
            bkd = "HEOS"
        else:
            bkd = backend
    
        # Create AbstractState
        self.fluid_as = AbstractState(bkd, fluid)


    def set_inlet_conditions(self, p_01, T_01):
        """
        Set the thermodynamic properties at the compressor suction.
        """

        if self.fluid is None:
            raise ValueError(
                "The 'fluid' attribute must be set before calling 'set_inlet_conditions'. "
                "Use 'set_fluid(fluid_name)' first.")

        self.T_01 = T_01
        self.p_01 = p_01
        self.fluid_as.update(PT_INPUTS, self.p_01, self.T_01)
        self.h_01 = self.fluid_as.hmass()
        self.s_01 = self.fluid_as.smass()
        self.rho_01 = self.fluid_as.rhomass()

    def set_operating_conditions(self, mdot, omega_rpm):
        """
        Set the compressor operating conditions.
        """
        self.mdot = mdot
        self.omega_rpm = omega_rpm
        self.omega_rads = 2 * math.pi * omega_rpm / 60
        
    def _properties(self):
        """
        Calculate properties

        Returns
        -------
        T, p, h, s, rho, a.

        """
        T = self.fluid_as.T()
        p = self.fluid_as.p()
        h = self.fluid_as.hmass()
        s = self.fluid_as.smass()
        rho = self.fluid_as.rhomass()
        a = self.fluid_as.speed_sound()
        
        return T, p, h, s, rho, a

    def slip_factor(self):
        # Reference:
        # (Wiesner, 1967) - doi:10.1115/1.3616734
        # (Meroni, 2018) - doi:10.1016/j.apenergy.2018.09.210
        # (Aungier, 2000) - doi:10.1115/1.800938
        sigma = 1 - math.sqrt(math.cos(math.radians(self.beta2_blade))) / self.N_blades_impeller ** 0.7
        sigma_star = math.sin(math.radians(19 + 0.2 * (90 - abs(self.beta2_blade))))
        A = (sigma - sigma_star) / (1 - sigma_star)
        B = (self.r1_rms / self.r2 - A) / (1 - A)
        if self.r1_rms / self.r2 < A:
            slip_factor = sigma
        else:
            slip_factor = sigma * (1 - B ** math.sqrt((90 - abs(self.beta2_blade)) / 10))
        return slip_factor

    def loss_inducer_incidence(self):
        # (Romei et al., 2020) - 10.1115/1.4046182
        f_inc = 0.7
        
        Dh_loss = (f_inc * self.w1_rms ** 2 *
                   (math.sin(
                       math.radians(abs(self.beta1_rms - self.beta1_blade_rms)))) ** 2)
        return Dh_loss

    def loss_impeller_skin_friction(self):
        # (Romei et al., 2020) - 10.1115/1.4046182

        D1_s = 2 * self.r1_shroud
        D1_h = 2 * self.r1_hub
        D_hydr = (math.pi * (D1_s ** 2 - D1_h ** 2) /
                  (2 * math.pi * D1_h + self.N_blades_impeller * (D1_s - D1_h)))
        
        # Compute blade-relative velocities at RMS
        c_blade_rms = self.u1_rms / math.tan(math.radians(abs(self.beta1_blade_rms)))
        c_blade_hub = c_blade_rms
        c_blade_shroud = c_blade_rms

        self.beta1_blade_hub = -math.degrees(math.atan(self.u1_hub / c_blade_hub))
        self.beta1_blade_shroud = -math.degrees(math.atan(self.u1_shroud / c_blade_shroud))

        cos_beta1_s = math.cos(math.radians(self.beta1_blade_shroud))
        cos_beta1_h = math.cos(math.radians(self.beta1_blade_hub))
        cos_beta2_bl = math.cos(math.radians(self.beta2_blade))
        Lb = (math.pi / 8 *
              (2 * self.r2 - (self.r1_shroud + self.r1_hub) - self.b2 + 2 * self.axial_length) *
              2 / ((cos_beta1_s + cos_beta1_h) / 2 + cos_beta2_bl))

        w1_rms_tangential = self.w1_rms * math.sin(abs(math.radians(self.beta1_rms)))
        W_ave = (self.c1_tangential + w1_rms_tangential + self.c2 + 2 * self.w1_hub + 3 * self.w2) / 8

        Dh_loss = 2 * self.friction_factor() * Lb / D_hydr * W_ave ** 2
        return Dh_loss

    def loss_impeller_blade_loading(self):
        # (Romei et al., 2020) - 10.1115/1.4046182

        # very similar to those used in
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210 and
        # (Oh et al., 1997) - 10.1243 / 0957650971537231

        F = (self.h_02 - self.h_01) / self.u2 ** 2 * self.w2 / self.w1_shroud
        E =  (self.N_blades_impeller / math.pi * (1 - self.r1_shroud / self.r2) +
              2 * self.r1_shroud / self.r2)
        Df = 1 - self.w2 / self.w1_rms + F / E
        Dh_loss = 0.05 * Df ** 2 * self.u2 ** 2
        return Dh_loss

    def loss_impeller_clearance(self):
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210
        # (Oh et al., 1997) - 10.1243/0957650971537231
        # (Romei et al., 2020) - 10.1115/1.4046182
        A = 0.6 * (self.radial_clearance / self.b2) * abs(self.c2_tangential)
        B = 4 * math.pi * (self.r1_shroud ** 2 - self.r1_hub ** 2) * abs(self.c2_tangential) * self.c1_meridional
        C = self.b2 * self.N_blades_impeller * (self.r2 - self.r1_shroud) * (1 + self.rho_2 / self.rho_1)
        Dh_loss = A * math.sqrt(B * C)
        return Dh_loss

    def loss_impeller_mixing(self):
        # (Romei et al., 2020) - 10.1115/1.4046182
        epsilon = 0.35
        tan_alpha2 = math.tan(math.radians(self.alpha2))
        Dh_loss =  0.5 * self.c2 ** 2 / (1 + tan_alpha2 ** 2) * (epsilon / (1 - epsilon)) ** 2
        return Dh_loss

    def loss_impeller_disk_friction(self):
        # (Romei et al., 2020) - 10.1115/1.4046182
        rho_ave = (self.rho_1 + self.rho_2) / 2

        Dh_loss = self.friction_factor() * rho_ave * self.r2 ** 2 * self.u2 ** 3 / (4 * self.mdot)
        return Dh_loss

    def loss_impeller_recirculation(self):
        # (Romei et al., 2020) - 10.1115/1.4046182

        # very similar to those used in
        # (Meroni et al., 2018) - 10.1016/j.apenergy.2018.09.210 and
        # (Oh et al., 1997) - 10.1243 / 0957650971537231
        F = (self.h_02 - self.h_01) / self.u2 ** 2 * self.w2 / self.w1_shroud
        E = (self.N_blades_impeller / math.pi * (1 - self.r1_shroud / self.r2) + 2 * self.r1_shroud / self.r2)
        Df = 1 - (self.w2 / self.w1_rms) + F / E
        Dh_loss = 8e-5 * math.sinh(3.5 * math.radians(self.alpha2) ** 3) * Df ** 2 * self.u2 ** 2
        return Dh_loss

    def loss_impeller_leakage(self):
        # (Romei et al., 2020) - 10.1115/1.4046182
        b_ave = (self.b1 + self.b2) / 2
        R_ave = (self.r2 + self.r1_shroud) / 2
        Dp_l = self.mdot * (self.r2 * self.c2_tangential - self.r1_shroud * self.c1_tangential) / (self.N_blades_impeller * R_ave * b_ave * self.axial_length)
        u_l = 0.816 * math.sqrt(2 * Dp_l / self.rho_2)
        mdot_l = self.rho_2 * u_l * self.radial_clearance * self.axial_length * self.N_blades_impeller
        Dh_loss = mdot_l * u_l * self.u2 / (2 * self.mdot)
        return Dh_loss

    def loss_vaned_diffuser_incidence(self):
        # (Romei et al., 2020) - 10.1115/1.4046182
        f_inc = 0.7 # values between 0.5 and 0.7
        self.alpha3 = math.degrees((math.acos(self.c3_meridional / self.c3)))
        Dh_loss = f_inc * self.c3 ** 2 * (
            math.sin(math.radians(self.alpha3) - math.radians(self.alpha3_blade)) ** 2)
        return Dh_loss

    def loss_vaned_diffuser_friction(self):
        # (Romei et al., 2020) - 10.1115/1.4046182

        c_ave = math.sqrt(self.c3 ** 2 + self.c4 ** 2) / 2

        cosa_alpha3_bl = math.cos(math.radians(self.alpha3_blade))
        cosa_alpha4_bl = math.cos(math.radians(self.alpha4_blade))

        L_vn = 2 * self.r3 * (self.r4 / self.r3 - 1) / (cosa_alpha3_bl + cosa_alpha4_bl)
        # vaned_diffuser_length = 22.8 * 1e-3

        A = math.pi * self.r3 / self.N_blades_diffuser - self.diff_blade_thickness_inlet
        D_3h = 2 * self.b3 * (2 * A) / (A + self.b4 / cosa_alpha3_bl)
        B = math.pi * self.r4 / self.N_blades_diffuser - self.diff_blade_thickness_outlet
        D_4h = 2 * self.b4 * (2 * B) / (B + self.b4 / cosa_alpha4_bl)
        D_hvn = (D_3h + D_4h) / 2

        delta_bl = 5.142 * self.friction_factor() * L_vn / D_hvn

        Dh_loss = 2 * self.friction_factor() * L_vn * c_ave ** 2 / (D_hvn * delta_bl ** 0.25)
        return Dh_loss

    def loss_factor_volute(self):
        # (Aungier, 2000) - doi:10.1115/1.800938
        # Meridional velocity loss

        c5_meridional = self.c5 * math.cos(math.radians(self.alpha5))
        loss_1 = (c5_meridional / self.c5) ** 2

        # Tangential velocity loss
        c5_tangential = self.c5 * math.sin(math.radians(self.alpha5))
        sp = self.r5 * c5_tangential / (self.r6 * self.c6)
        if sp >= 1:
            loss_2 = 0.5 * self.r5 / self.r6 * (c5_tangential / self.c5) ** 2 * (1 - 1 / sp ** 2)
        else:
            loss_2 = self.r5 / self.r6 * (c5_tangential / self.c5) ** 2 * (1 - 1 / sp) ** 2

        # Skin friction losses
        L_ave = math.pi * (self.r5 + self.r6) / 2  # (m) - Average length travelled by the fluid in the volute
        D_hydr = math.sqrt(4 * self.A6 / math.pi)
        self.fluid_as.update(DmassT_INPUTS, self.rho_6, self.T_6)
        mu_6 = self.fluid_as.viscosity()
        Re = (self.rho_6 * self.c6 * D_hydr) / mu_6
        Re_e = (Re - 2000) * (self.roughness / D_hydr)

        loss_3 = 4 * self.friction_factor() * (self.c6 / self.c5) ** 2 * L_ave / D_hydr

        return loss_1 + loss_2 + loss_3  # Total pressure loss factor
    
    def loss_volute(self):
        # (Romei et al., 2020) - 10.1115/1.4046182
                
        # Meridional velocity loss
        dh_loss = self.c4_meridional ** 2 / 2
        return dh_loss      

    def _friction_factor_haaland(self, Re, D_hydr):
        eD = self.roughness / D_hydr
        if Re <= 0:
            raise ValueError("Reynolds number must be positive.")
        return 1 / (-1.8 * np.log10((6.9 / Re) + (eD / 3.7) ** 1.11)) ** 2

    def _friction_factor_colebrook(self, Re, Re_e, D_hydr):
        """
        Calculate the Darcy friction factor using a Colebrook-like formulation
        and robust bounded root-finding via Brent's method (brentq).

        Reference: Aungier (2000), pp. 74–75
        """

        # Colebrook function - CFS (smooth pipe)
        def f0_cfs(x):
            return 2.51 * 10 ** (1 / (2 * x)) - Re * x

        # Colebrook function - CFR (rough pipe)
        def f0_cfr(x):
            return 10 ** (1 / (2 * x)) * self.roughness - 3.71 * D_hydr

        try:
            cfs_root = brentq(f0_cfs, 0.05, 1.5, xtol=1e-6)
            cfs = cfs_root ** 2 / 4
        except ValueError:
            print("Warning: brentq failed for cfs; using fallback")
            cfs = 0.02  # reasonable fallback

        try:
            cfr_root = brentq(f0_cfr, 0.05, 1.5, xtol=1e-6)
            cfr = cfr_root ** 2 / 4
        except ValueError:
            print("Warning: brentq failed for cfr; using fallback")
            cfr = 0.02 # reasonable fallback

        # Transition between CFS and CFR
        if Re_e < 60:
            cf = cfs
        else:
            cf = cfs + (cfr - cfs) * (1 - 60 / Re_e)

        return cf


    def friction_factor(self, method="constant", Re=None, Re_e=None, D_hydr=None):
            """
            Parameters
            ----------
            method : str
                One of {"colebrook", "haaland", "constant"}.
                  - "colebrook": Colebrook-like with Brent's method and smooth/rough transition.
                               Needs Re, Re_e, D_hydr.
                  - "haaland": Haaland explicit correlation.
                               Needs Re, D_hydr.
                  - "constant": Returns 0.006 (no inputs required).
            Re : float, optional
                Reynolds number (required for "colebrook" and "haaland").
            Re_e : float, optional
                Equivalent roughness Reynolds number (required for "colebrook").
            D_hydr : float, optional
                Hydraulic diameter (required for "colebrook" and "haaland").

            Returns
            -------
            float
                Darcy friction factor.
            """
            m = method.lower()
            if m == "constant":
                cf = 0.006 # (Romei et al., 2020) - 10.1115/1.4046182
                cf_mod = cf * 1 # Increased to match experimental data
                return cf_mod

            if m == "haaland":
                if Re is None or D_hydr is None:
                    raise ValueError("Haaland method requires Re and D_hydr.")
                return self._friction_factor_haaland(Re, D_hydr)

            if m == "colebrook":
                if Re is None or Re_e is None or D_hydr is None:
                    raise ValueError("colebrook method requires Re, Re_e, and D_hydr.")
                return self._friction_factor_colebrook(Re, Re_e, D_hydr)

            raise ValueError('Unknown method. Use "aungier", "haaland", or "constant".')

    def _impeller_losses(self):
        # internal losses
        self.Dh_loss_inducer_incidence = self.loss_inducer_incidence()
        self.Dh_loss_impeller_skin_friction = self.loss_impeller_skin_friction()
        self.Dh_loss_impeller_blade_loading = self.loss_impeller_blade_loading()
        self.Dh_loss_impeller_clearance = self.loss_impeller_clearance()
        self.Dh_loss_impeller_mixing = self.loss_impeller_mixing()

        self.Dh_internal_loss = (self.Dh_loss_inducer_incidence +
                                 self.Dh_loss_impeller_skin_friction +
                                 self.Dh_loss_impeller_blade_loading +
                                 self.Dh_loss_impeller_clearance +
                                 self.Dh_loss_impeller_mixing)

        # External losses
        self.Dh_loss_impeller_disk_friction = self.loss_impeller_disk_friction()
        self.Dh_loss_impeller_recirculation = self.loss_impeller_recirculation()
        self.Dh_loss_impeller_leakage = self.loss_impeller_leakage()

        self.Dh_external_loss = (self.Dh_loss_impeller_disk_friction +
                                 self.Dh_loss_impeller_recirculation +
                                 self.Dh_loss_impeller_leakage)
        
    def simulate(self, verbose=True):
        """
        Begin simulation of the compressor by calculating inlet velocity and static conditions.

        Parameters:
        verbose (bool): If True, prints simulation status and key parameters.
        """
        self.verbose = verbose

        if self.verbose:
            print("Running compressor simulation...")
            print(f"Mass flow rate: {self.mdot} kg/s")
            print(f"Rotational speed: {self.omega_rpm} RPM ({self.omega_rads:.2f} rad/s)")
            print(f"Inlet total pressure: {self.p_01:.2f} Pa")
            print(f"Inlet total temperature: {self.T_01:.2f} K")
            print(f"Fluid: {self.fluid}")

        """ Impeller inlet """

        self.u1_rms = self.omega_rads * self.r1_rms
        self.u1_hub = self.omega_rads * self.r1_hub
        self.u1_shroud = self.omega_rads * self.r1_shroud
        
        def section_1(rho_guess):
            self.rho_1 = rho_guess
            
            # Velocities
            self.c1_meridional = self.mdot / (self.A1 * self.rho_1)
            self.c1 = self.c1_meridional / math.cos(math.radians(self.alpha1))
            self.c1_tangential = self.c1 * math.sin(math.radians(self.alpha1))
            
            self.w1_rms = math.sqrt(self.c1 ** 2 + self.u1_rms ** 2)
            self.beta1_rms = - math.degrees(math.acos(self.c1 / self.w1_rms))
            
            self.w1_shroud = math.sqrt(self.c1 ** 2 + self.u1_shroud ** 2)
            self.beta1_shroud = - math.degrees(math.acos(self.c1 / self.w1_shroud))
            
            self.w1_hub = math.sqrt(self.c1 ** 2 + self.u1_hub ** 2)
            self.beta1_hub = - math.degrees(math.acos(self.c1 / self.w1_hub))
            
            # Thermodynamic properties
            self.h_1 = self.h_01 - self.c1 ** 2 / 2
            self.s_1 = self.s_01
            self.fluid_as.update(HmassSmass_INPUTS, self.h_1, self.s_1)
         
        def section_1_residual(rho_guess):
            # Section calculation
            section_1(rho_guess)
            #residual            
            res = rho_guess - self.fluid_as.rhomass()
            return res
        
        self.convergence_1 = root(section_1_residual, self.rho_01, tol=1e-6)
        
        # Final update
        if self.convergence_1.success:
            section_1(float(self.convergence_1.x))
            if self.verbose:
                print("Warning: fsolve converged for impeller inlet")
        else:
            section_1(self.rho_01) # Fall back
            if self.verbose:
                print("Warning: fsolve did not converge for impeller inlet")   
        
        # Update of thermodynamic properties
        self.T_1, self.p_1, self.h_1, self.s_1, self.rho_1, a1 = self._properties()
        self.Ma_1 = self.c1 / a1

        """ Impeller outlet """
      
        self.u2 = self.omega_rads * self.r2  
        
        def section_2(rho_guess):
            self.rho_2 = rho_guess

            # Velocities
            self.c2_meridional = self.mdot / (self.rho_2 * self.A2)
            tan_beta2_bl = math.tan(math.radians(abs(self.beta2_blade)))
            self.c2_tangential = self.slip_factor() * self.u2 - self.c2_meridional * tan_beta2_bl
            self.c2 = math.sqrt(self.c2_meridional ** 2 + self.c2_tangential ** 2)

            self.w2_tangential = self.u2 - self.c2_tangential
            self.w2 = math.sqrt(self.w2_tangential ** 2 + self.c2_meridional ** 2)
            self.alpha2 = math.degrees(math.acos(self.c2_meridional / self.c2))
            self.beta2 = math.degrees(math.acos(self.c2_meridional / self.w2))
            
            # Thermodynamic properties
            self.L_eulero = self.u2 * self.c2_tangential - self.u1_rms * self.c1_tangential
            self.h_02 = self.h_01 + self.L_eulero
            self.h_2 = self.h_02 - self.c2 ** 2 / 2
            
            # (Romei et al., 2020) - 10.1115/1.4046182
            self.axial_length = 2 * self.r2 * (
                    0.014 + 0.023 * self.r2 / self.r1_hub +
                    2.012 * self.mdot / self.rho_01 / self.u2 / (2 * self.r2) ** 2) # Calculated before the losses

            # Impeller losses
            self._impeller_losses()

            # Isentropic outlet conditions (to calculate pressure)
            h_2is = self.h_02 - self.c2 ** 2 / 2 - self.Dh_internal_loss
            self.fluid_as.update(HmassSmass_INPUTS, h_2is, self.s_1)
            self.p_2 = self.fluid_as.p()
            
            # Actual outlet conditions
            self.fluid_as.update(HmassP_INPUTS, self.h_2, self.p_2)

        def section_2_residual(rho_guess):
            # Evaluate the section
            section_2(rho_guess)
            # Compare guess and model result      
            res =  rho_guess - self.fluid_as.rhomass()
            return res

        self.convergence_2 = root(section_2_residual, self.rho_1, tol=1e-6)
        
        # Final update
        if self.convergence_2.success:
            section_2(float(self.convergence_2.x))
            if self.verbose:
                print("Warning: fsolve converged for impeller outlet")
        else:
            section_2(self.rho_1) # Fall back
            if self.verbose:
                print("Warning: fsolve did not converge for impeller outlet")   

        # Update of thermodynamic properties
        self.T_2, self.p_2, self.h_2, self.s_2, self.rho_2, a2 = self._properties()
        self.Ma_2 = self.c2 / a2
        

        """ Vaned diffuser Inlet"""

        # self.h_03 = self.h_02 + self.Dh_external_loss # The external losses are added to the fluid total enthalpy after the impeller outlet
        self.h_03 = self.h_02

        def section_3(rho_guess):
            self.rho_3 = rho_guess
            
            # Due to the thickness of vaned diffuser blades the fluid accelerates
            # While the meridional component increases, the tangential one
            # is assumed to be the same as the one at the impeller outlet
            
            # Velocities
            self.c3_meridional = self.mdot / (self.rho_3 * self.A3)
            self.c3_tangential = self.c2_tangential
            self.c3 = math.sqrt(self.c3_meridional ** 2 + self.c3_tangential ** 2)
            
            # Thermodynamic properties
            self.h_3 = self.h_03 - self.c3 ** 2 / 2
            # Isentropic transformation in the vaneless cavity
            # (Meroni - DOI:10.1016/j.apenergy.2018.09.210 - Eq. 20)
            self.s_3 = self.s_2
            self.fluid_as.update(HmassSmass_INPUTS, self.h_3, self.s_3)

        def section_3_residual(rho_guess):
            # Evaluate the section
            section_3(rho_guess)
            # Compare guess and model result      
            res =  rho_guess - self.fluid_as.rhomass()
            return res

        self.convergence_3 = root(section_3_residual, self.rho_2, tol=1e-6)
         
        # Final update
        if self.convergence_3.success:
            section_3(float(self.convergence_3.x))
            if self.verbose:
                 print("Warning: fsolve converged for vaned diffuser inlet")
        else:
            section_3(self.rho_2) # Fall back
            if self.verbose:
                print("Warning: fsolve did not converge for vaned diffuser inlet")   

        # Update of thermodynamic properties
        self.T_3, self.p_3, self.h_3, self.s_3, self.rho_3, a3 = self._properties()
        self.Ma_3 = self.c3 / a3

        """ Vaned Diffuser Outlet"""

        self.h_04 = self.h_03
        
        def section_4(rho_guess):
            self.rho_4 = rho_guess
            
            # Velocities
            self.alpha4 = self.alpha4_blade  # (Meroni - DOI:10.1016/j.apenergy.2018.09.210 - page 145)
            
            self.c4_meridional = self.mdot / (self.rho_4 * self.A4) 
            self.c4 = self.c4_meridional / math.cos(math.radians(self.alpha4))
            self.c4_tangential = self.c4 * math.sin(math.radians(self.alpha4))
            
            # Thermodynamic properties
            self.h_4 = self.h_04 - self.c4 ** 2 / 2
            
            # Vaned diffuser losses
            self.Dh_loss_vaned_diffuser_incidence = self.loss_vaned_diffuser_incidence()
            self.Dh_loss_vaned_diffuser_friction = self.loss_vaned_diffuser_friction()
            
            self.Dh_vaned_diffuser = self.Dh_loss_vaned_diffuser_incidence + self.Dh_loss_vaned_diffuser_friction
            
            # Isentropic outlet conditions (to calculate pressure)           
            h_4is = self.h_4 - self.Dh_vaned_diffuser # (Meroni - DOI:10.1016/j.apenergy.2018.09.210 - Eq. 23)
            self.fluid_as.update(HmassSmass_INPUTS, h_4is, self.s_3)
            self.p_4 = self.fluid_as.p()
            
            # Actual outlet conditions
            self.fluid_as.update(HmassP_INPUTS, self.h_4, self.p_4)
            
        
        def section_4_residual(rho_guess):
            # Evaluate the section
            section_4(rho_guess)
            # Compare guess and model result
            res =  rho_guess - self.fluid_as.rhomass()
            return res

        # Solve for rho_4
        self.convergence_4 = root(section_4_residual, self.rho_3, tol=1e-6)

        # Final update
        if self.convergence_4.success:
            section_4(float(self.convergence_4.x))
            if self.verbose:
                 print("Warning: fsolve converged for vaned diffuser outlet")
        else:
            section_4(self.rho_3) # Fall back
            if self.verbose:
                print("Warning: fsolve did not converge for vaned diffuser outlet")   

        # Update of thermodynamic properties
        self.T_4, self.p_4, self.h_4, self.s_4, self.rho_4, a4 = self._properties()
        self.Ma_4 = self.c4 / a4
        
        # Total properties
        self.fluid_as.update(HmassSmass_INPUTS, self.h_04, self.s_4)
        self.p_04 = self.fluid_as.p()

        """ Volute outlet """

        self.h_05 = self.h_04  # total enthalpy conservation

        def section_5(rho_guess):
            self.rho_5 = rho_guess
            
            # Velocities
            self.c5 = self.mdot / (self.rho_5 * self.A5)
            
            # Thermodynamic properties
            self.h_5 = self.h_05 - self.c5 ** 2 / 2
            
            # Volute losses
            self.Dh_loss_volute = self.loss_volute()
            
            # Isentropic outlet conditions (to calculate pressure)   
            h_5is = self.h_5 - self.Dh_loss_volute
            self.fluid_as.update(HmassSmass_INPUTS, h_5is, self.s_4)
            self.p_5 = self.fluid_as.p()
            
            # Actual outlet conditions
            self.fluid_as.update(HmassP_INPUTS, self.h_5, self.p_5)
            
        def section_5_residual(rho_guess):
            # Evaluate the section
            section_5(rho_guess)
            # Compare guess and model result
            res =  rho_guess - self.fluid_as.rhomass()
            return res

        # Solve for rho_5
        self.convergence_5 = root(section_5_residual, self.rho_4, tol=1e-6)

        # Final update
        if self.convergence_5.success:
            section_5(float(self.convergence_5.x))
            if self.verbose:
                 print("Warning: fsolve converged for vaned diffuser outlet")
        else:
            section_5(self.rho_4) # Fall back
            if self.verbose:
                print("Warning: fsolve did not converge for vaned diffuser outlet")   

        # Update of thermodynamic properties
        self.T_5, self.p_5, self.h_5, self.s_5, self.rho_5, a5 = self._properties()
        self.Ma_5 = self.c5 / a5
        
        # Total properties
        self.fluid_as.update(HmassSmass_INPUTS, self.h_05, self.s_5)
        self.p_05 = self.fluid_as.p()
        self.T_05 = self.fluid_as.T()

        """
        Global Performance
        """
        # Efficiency
        
        # Ideal work input
        Dh0_id = (self.L_eulero
                  - self.Dh_internal_loss
                  - self.Dh_vaned_diffuser
                  - self.Dh_loss_volute)
        
        # Actual work input
        Dh0_act = self.L_eulero + self.Dh_external_loss
        self.W = Dh0_act # Total work input
        self.Wdot = self.mdot * Dh0_act # Total power input
        
        # Efficiency (isentropic)
        self.eta_is_tt = Dh0_id / Dh0_act # Total-total
        self.eta_is_ts = (Dh0_id - self.c5 ** 2 / 2) / Dh0_act # Total-static
        
        # Efficiency (politropic)
        Dh0 = self.h_05 - self.h_01
        Ds = self.s_5 - self.s_1
        DT = self.T_5 - self.T_1
        Dh0_pol = Dh0 - Ds * DT / math.log(self.T_5 / self.T_1)     # Mallen-Saville method - (Casey, 2021 - ISBN: 978-1-108-41667-2)
        self.eta_pol_tt = Dh0_pol / Dh0_act
        self.eta_pol_ts = (Dh0_pol - self.c5 ** 2 / 2) / Dh0_act

        self.phi = self.mdot / (self.rho_01 * self.u2 * (2 * self.r2) ** 2) # Flow coefficienct
        self.psi = Dh0_act / self.u2 ** 2 # Work coefficienct
        self.psi_eu = self.L_eulero / self.u2 ** 2  # Work coefficienct
        
        # Pressure ratio
        self.PR_tt = self.p_05 / self.p_01 # Total-total
        self.PR_ts = self.p_5 / self.p_01 # Total-static
        self.PR_ss = self.p_5 / self.p_1 # Stati-static
        
        self.RR = (self.h_2 + self.w2 ** 2 / 2 - self.u2 ** 2 / 2) - (
                self.h_1 + self.w1_rms ** 2 / 2 - self.u1_rms ** 2 / 2) # Rothalpy
        
        
   
    def states_dataframe(self):
        """Return a DataFrame of compressor internal static states."""
        data = {
            "Section": ["Impeller Inlet (1)", "Impeller Outlet (2)",
                        "Vaned Diffuser Inlet (4)",
                        "Vaned Diffuser Outlet (5)", "Volute Exit (6)"],
            "p [Pa]":       [self.p_1, self.p_2, self.p_3, self.p_4, self.p_5],
            "T [K]":        [self.T_1, self.T_2, self.T_3, self.T_4, self.T_5],
            "h [J/kg]":     [self.h_1, self.h_2, self.h_3, self.h_4, self.h_5],
            "s [J/kg/K]":   [self.s_1, self.s_2, self.s_3, self.s_4, self.s_5],
            "rho [kg/m3]":  [self.rho_1, self.rho_2, self.rho_3, self.rho_4, self.rho_5],
            "C [m/s]":      [self.c1, self.c2, self.c3, self.c4, self.c5],
        }
        return pd.DataFrame(data)

    def export_to_excel(self, filename=None):
        """Export static thermodynamic properties to Excel."""
        if filename is None:
            filename = "compressor_static_data.xlsx"
        df = self.states_dataframe()
        df.to_excel(filename, index=False)

    def export_losses_dataframe(self):
        """
        Return a pandas DataFrame of all identified loss mechanisms.
        """
        data = {
            'Loss Mechanism': [
                'Inducer Incidence',
                'Impeller Skin Friction',
                'Impeller Blade Loading',
                'Impeller Clearance',
                'Impeller Mixing',
                'Impeller Disk Friction',
                'Impeller Recirculation',
                'Impeller Leakage',
                'Vaned Diffuser Incidence',
                'Vaned Diffuser Friction',
                'Volute',
                ],
            'Loss [J/kg]': [
                self.Dh_loss_inducer_incidence,
                self.Dh_loss_impeller_skin_friction,
                self.Dh_loss_impeller_blade_loading,
                self.Dh_loss_impeller_clearance,
                self.Dh_loss_impeller_mixing,
                self.Dh_loss_impeller_disk_friction,
                self.Dh_loss_impeller_recirculation,
                self.Dh_loss_impeller_leakage,
                self.Dh_loss_vaned_diffuser_incidence,
                self.Dh_loss_vaned_diffuser_friction,
                self.Dh_loss_volute,
                ]
            }

        return pd.DataFrame(data)