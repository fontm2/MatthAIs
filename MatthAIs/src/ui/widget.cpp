#include "widget.h"
#include "ui_widget.h"

using namespace std;

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    this->setWindowTitle("MatthAIs");

    // Set userinteractions
    ui->qcpwidget->setInteraction(QCP::iRangeDrag, true);
    ui->qcpwidget->setInteraction(QCP::iRangeZoom, true);
    ui->qcpwidget->setInteraction(QCP::iSelectAxes, true);
    ui->qcpwidget->setInteraction(QCP::iSelectOther, true);
    QObject::connect(ui->qcpwidget, SIGNAL (mouseDoubleClick (QMouseEvent *)), ui->qcpwidget,SLOT (rescaleAxes()));
    // set some Axis Properties
    ui->qcpwidget->xAxis->setLabel("Epoche");
    ui->qcpwidget->yAxis->setLabel("Error");
    ui->qcpwidget->xAxis->grid()->setSubGridVisible(true);

    // Add Graph and set some properties
    // setup for graph 0: key axis bottom, value axis left (those are the default axes)
    ui->qcpwidget->addGraph(ui->qcpwidget->xAxis, ui->qcpwidget->yAxis);
    ui->qcpwidget->graph(0)->setPen(QPen(Qt::blue)); // line color blue for first graph
    ui->qcpwidget->graph(0)->setName("Errorplot");

    // Register data type for custom Signal "newData"
    qRegisterMetaType<QVector<double>>("QVector<double>");
    qRegisterMetaType<QVector<QPoint>>("QVector<QPoint>");

    //connects
    connect(ui->run, SIGNAL(clicked()), this, SLOT(run_clicked()));
    connect(this,SIGNAL(newData(QVector<double>,QVector<double>,QVector<QPoint>,QVector<QPoint>)),this,SLOT(setPlotData(QVector<double>,QVector<double>,QVector<QPoint>,QVector<QPoint>)));




    //initialize network
//    vector<int> layerdims{10,2,1};
//    vector<ActivationFunction> layeractfunc{ActivationFunction::ReLU,ActivationFunction::Sigmoid, ActivationFunction::Sigmoid};
//    MatrixXd Input = net.load_csv<MatrixXd>("Patterns.txt");
//    net.init(Input, layerdims, layeractfunc, 1, 0.001);

    vector<int> layerdims{15,2,1};
    vector<ActivationFunction> layeractfunc{ActivationFunction::Sigmoid,ActivationFunction::Sigmoid, ActivationFunction::Sigmoid};
    MatrixXd Input = net.load_csv<MatrixXd>("facedata.txt");
    //normalize except th labels
    Input.block(0,0,Input.rows(),Input.cols()-1) = Input.block(0,0,Input.rows(),Input.cols()-1)/255;
    net.init(Input.block(0,0,100,Input.cols()), layerdims, layeractfunc, 1, 0.01);

}

Widget::~Widget()
{
    delete ui;
}

void Widget::setPlotData(QVector<double> x, QVector<double> y, QVector<QPoint> h_plane, QVector<QPoint> p)
{
    QPixmap pixmap(306,306);
    pixmap.fill(QColor("transparent"));
    QPainter painter(&pixmap);
    painter.setPen(QPen(Qt::black, 3, Qt::SolidLine, Qt::SquareCap));
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(0, 0, 304, 304);
    painter.setPen(QPen(Qt::darkGreen, 2));
    painter.drawLine(h_plane[0], h_plane[1]);

    //negative examples
    painter.setPen(QPen(Qt::red, 5));
    for(int i = 0; i < p.size()/2; i++)
    {
        painter.drawPoint(p[i]);
    }
    //positive examples
    painter.setPen(QPen(Qt::blue, 5));
    for(int i = p.size()/2; i < p.size(); i++)
    {
        painter.drawPoint(p[i]);
    }
    ui->drawinglabel->setPixmap(pixmap);

    ui->qcpwidget->graph(0)->setData(x,y);
    ui->qcpwidget->rescaleAxes();
    ui->qcpwidget->replot();
}

void Widget::run_clicked()
{
    train_thread = std::thread(&Widget::run, this);
}

void Widget::run()
{
    for(int i = 0; i < 1000; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mu);
            errors.push_back(net.train());
            epoche.push_back(i+1);

            Hyperplane.clear();
            points.clear();
            double intercept = -net.Hyperplanes[2]/net.Hyperplanes[1];//-bias_weight/w2
            double slope = -net.Hyperplanes[0]/net.Hyperplanes[1];// -w1/w2
            QPoint left;
            left.setX(0);
            left.setY((int)(intercept * 306));
            QPoint right;
            right.setX(306);
            right.setY((int)(intercept*306 + slope*306));
            Hyperplane.push_back(left);
            Hyperplane.push_back(right);
            for(int i = 0; i <net.PointsToDraw.size(); i++)
            {
                QPoint p;
                p.setX(net.PointsToDraw[i].first*300);
                p.setY(net.PointsToDraw[i].second*300);
                points.push_back(p);
            }


            //std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
       emit newData(QVector <double>::fromStdVector(epoche), QVector <double>::fromStdVector(errors), QVector <QPoint>::fromStdVector(Hyperplane), QVector <QPoint>::fromStdVector(points));
    }
}

