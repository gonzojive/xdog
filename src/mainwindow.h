//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
#pragma once

#include "ui_mainwindow.h"
#include "imageview.h"
#include "videoplayer.h"
#include "cpu_image.h"
#include "gpu_image.h"
#include "param.h"
#include "paramui.h"

class MainWindow : public QMainWindow, protected Ui_MainWindow, public ImageView::Handler {
    Q_OBJECT
public:
    MainWindow();
    ~MainWindow();

    void restoreAppState();
    void saveAppState();
    bool event(QEvent *event);

    const QImage& image() const { return m_result[m_select->currentIndex()]; }

protected slots:
    void on_actionOpen_triggered();
    void on_actionAbout_triggered();
    void on_actionSelectDevice_triggered();
    void on_actionRecord_triggered();
    void on_actionSavePNG_triggered();
    void on_actionLoadSettings_triggered();
    void on_actionSaveSettings_triggered();
    void on_actionShowSettings_triggered();

    void setDirty();
    void process();
    
    void onIndexChanged(int);
    void onVideoChanged(int nframes);

signals:
    void imageChanged(const QImage&);

protected:
    VideoPlayer *m_player;
    ParamUI *m_paramui;
    cpu_image<float4> m_st;
    cpu_image<float4> m_tfm;
    gpu_image<float> m_noise;
    QImage m_result[3];
    bool m_dirty;

    QString output;
    QString input_gamma;
    QString st_type;
    double sigma_c;
    double precision_sigma_c;
    int etf_N;
    
    bool enable_bf;
    QString filter_type;
    int n_a; 
    int n_e; 
    double sigma_dg;
    double sigma_rg;
    double sigma_dt;
    double sigma_rt;
    double bf_alpha;
    double precision_g;
    double precision_t;

    QString dog_type;
    double sigma_e;
    double precision_e;
    double dog_k;
    double sigma_m;
    double precision_m;
    double step_m;
    QString dog_adj_func;
    bool dog_reparam;
    double dog_tau;
    double dog_eps;
    double dog_phi;
    double dog_p;
    double dog_eps_p;
    double dog_phi_p;
    QString dog_fgauss;

    ParamGroup* dog_tau_g;
    ParamDouble* dog_tau_ptr;
    ParamDouble* dog_eps_ptr;
    ParamDouble* dog_phi_ptr;
    ParamGroup* dog_p_g;
    ParamDouble* dog_p_ptr;
    ParamDouble* dog_eps_p_ptr;
    ParamDouble* dog_phi_p_ptr;

    bool quantization;
    QString quant_type;
    int nbins; 
    double phi_q;
    double lambda_delta;
    double omega_delta;
    double lambda_phi;
    double omega_phi;
    bool warp_sharp;
    double sigma_w;
    double precision_w;
    double phi_w;
    bool final_smooth;
    QString final_type;
    double sigma_a;

protected slots:
    void dogChanged();
};
