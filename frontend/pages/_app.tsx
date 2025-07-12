import '../styles/globals.css';
import Layout from '../components/Layout';
import { ToastProvider } from '../components/Toast';
import React from 'react';
import type { AppProps } from 'next/app';
import { ThemeProvider } from 'next-themes';

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <ToastProvider>
        <Layout>
          <Component {...pageProps} />
        </Layout>
      </ToastProvider>
    </ThemeProvider>
  );
}
