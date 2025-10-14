// src/theme.js
import { createTheme, alpha } from "@mui/material/styles";

const mk = (opts) => {
  const {
    primary, secondary, bg, paper, mode = "light",
    text, border = alpha("#000", 0.06), appBg
  } = opts;

  const t = createTheme({
    palette: {
      mode,
      primary: { main: primary },
      secondary: { main: secondary },
      text: {
        primary: text?.primary ?? (mode === "light" ? "#111827" : "#E5E7EB"),
        secondary: text?.secondary ?? (mode === "light" ? "#6B7280" : "#94A3B8"),
      },
      background: { default: bg, paper },
    },
    shape: { borderRadius: 16 },
    typography: {
      fontFamily: '"Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif',
      h6: { fontWeight: 800, letterSpacing: -0.2 },
      body2: { lineHeight: 1.65, fontSize: 14.5 },
      caption: { color: "#64748B" },
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: 18,
            border: `1px solid ${border}`,
            backdropFilter: "blur(6px)",
          },
        },
      },
      MuiButton: {
        defaultProps: { disableElevation: true },
        styleOverrides: {
          root: { borderRadius: 999, textTransform: "none", fontWeight: 600, paddingInline: 18 },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: { borderRadius: 999, backgroundColor: alpha(primary, 0.08) },
        },
      },
      MuiTextField: { styleOverrides: { root: { borderRadius: 12, backgroundColor: paper } } },
    },
  });

  // custom tokens
  t.custom = {
    appBg,
    cardShadow: "0 10px 30px " + alpha("#000", 0.06),
    border,
  };
  return t;
};

// Polished presets
export const PRESETS = {
  indigo: mk({
    primary: "#4F46E5",
    secondary: "#10B981",
    bg: "#FFFFFF",
    paper: "#FFFFFF",
    appBg:
      "radial-gradient(1200px 600px at -10% -20%, #EEF2FF 0%, rgba(238,242,255,0) 60%), radial-gradient(1000px 500px at 110% 120%, #E0E7FF 0%, rgba(224,231,255,0) 60%), #F7F8FB",
    border: alpha("#0F172A", 0.08),
  }),
  sapphire: mk({
    primary: "#2563EB",
    secondary: "#06B6D4",
    bg: "#FFFFFF",
    paper: "#FFFFFF",
    appBg: "linear-gradient(180deg, #F1F5FF 0%, #F7FAFF 40%, #F7F8FB 100%)",
    border: alpha("#0B1324", 0.08),
  }),
  charcoal: mk({
    primary: "#111827",
    secondary: "#4B5563",
    bg: "#FFFFFF",
    paper: "#FFFFFF",
    appBg: "linear-gradient(180deg, #F6F7FB 0%, #EFF1F5 100%)",
    border: alpha("#111827", 0.12),
  }),
  warm: mk({
    primary: "#8B5CF6",
    secondary: "#F59E0B",
    bg: "#FFFFFF",
    paper: "#FFFFFF",
    appBg:
      "radial-gradient(800px 400px at 0% 0%, #FFF7ED 0%, rgba(255,247,237,0) 60%), #F9FAFB",
    border: alpha("#92400E", 0.10),
  }),
  dark: mk({
    mode: "dark",
    primary: "#7C83FF",
    secondary: "#06B6D4",
    bg: "#0B1020",
    paper: "#0F172A",
    appBg:
      "radial-gradient(900px 500px at -10% -10%, rgba(39,62,160,0.35) 0%, rgba(39,62,160,0) 60%), radial-gradient(900px 500px at 110% 110%, rgba(12,148,161,0.25) 0%, rgba(12,148,161,0) 60%), #0B1020",
    border: alpha("#93C5FD", 0.15),
  }),
};

// factory
export const makeTheme = (name = "indigo") => PRESETS[name] ?? PRESETS.indigo;
