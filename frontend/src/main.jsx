import React, { useMemo, useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import { ThemeProvider, CssBaseline } from "@mui/material";
import { makeTheme } from "./theme.js";
import "@fontsource/inter";

function Root() {
  const [preset, setPreset] = useState(localStorage.getItem("themePreset") || "indigo");
  const theme = useMemo(() => makeTheme(preset), [preset]);

  useEffect(() => localStorage.setItem("themePreset", preset), [preset]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App preset={preset} setPreset={setPreset} />
    </ThemeProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<Root />);
