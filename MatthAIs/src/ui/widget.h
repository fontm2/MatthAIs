#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <iterator>     // std::ostream_iterator
#include <string.h>
#include <fstream>      //file stream for reading .txt files
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <thread>
#include <network.h>
#include <mutex>

using namespace std;

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    mutex mu;
    Network net;
    explicit Widget(QWidget *parent = 0);
    ~Widget();
    void run();

private slots:
    void run_clicked();
    void setPlotData(QVector<double> x,QVector<double> y, QVector<QPoint> h_plane, QVector<QPoint> p);
signals:
    void newData(QVector<double> x, QVector<double> y, QVector<QPoint> h_plane, QVector<QPoint> p);
private:
    Ui::Widget *ui;
    vector<double> errors;
    vector<double> epoche;
    std::thread train_thread;
    vector<QPoint> Hyperplane;
    vector<QPoint> points;

};

#endif // WIDGET_H
