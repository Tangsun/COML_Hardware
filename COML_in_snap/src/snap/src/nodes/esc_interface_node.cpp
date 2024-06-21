/**
 * @file esc_interface_node.cpp
 * @brief Node to software arm/disarm esc interface via ROS service call
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 July 2019
 */

#include <string>
#include <cstring>
#include <csignal>
#include <fstream>

#include <ros/ros.h>

#include <ncurses.h>

#include <sensor_msgs/BatteryState.h>
#include <std_srvs/SetBool.h>
#include <std_srvs/Trigger.h>

// to be set by sighandler
volatile sig_atomic_t stop = 0;
void handle_sigint(int s) { stop = true; }

namespace acl {
namespace snap {

class ESCInterfaceNode
{
public:
    ESCInterfaceNode(const ros::NodeHandle& nh)
    : nh_(nh)
    {
        sub_battery_ = nh_.subscribe("battery", 1, &ESCInterfaceNode::battCb, this);

        // create service clients that will be used
        srv_arm_ = nh_.serviceClient<std_srvs::SetBool>("snap/arm");
        srv_isarmed_ = nh_.serviceClient<std_srvs::Trigger>("snap/is_armed");

        // determine vehicle name from the namespace
        name_ = ros::this_node::getNamespace();
        size_t n = name_.find_first_not_of('/');
        name_.erase(0, n); // remove leading slash

        // wait for services to exist, and then query arm state
        waitForServices();

        // determine if this is an sfpro or not
        ros::param::param<bool>("snap/sfpro", sfpro_, false);

        // initialized ncurses terminal ui
        initTUI();

        // display if we are armed or not
        updateTUI();
    }

    ~ESCInterfaceNode() = default;

    void spin()
    {
        // timeout for getch (ms)
        timeout(1000);

        while (ros::ok() && !stop) {
            int ch = getch();

            // clear the state of the screen from last time
            clear();

            if (ch == ' ') {
                if (armed_) disarm();
                else arm();
            } else if (ch == ERR) {
                if (!srv_arm_.exists() || !srv_isarmed_.exists()) {
                    mvprintw(0,0,"Disconnected from snap stack, waiting...");
                    refresh();
                    waitForServices();
                }
            }

            float cputemp = getCPUTemp();
            updateTUI(cputemp);

            ros::spinOnce();
        }

        closeTUI();

        // disarm on the way out
        ROS_INFO("Disarming on quit");
        disarm();
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_battery_;
    ros::ServiceClient srv_arm_, srv_isarmed_;
    std::string name_; ///< vehicle name to display

    double voltage_ = 0, current_ = 0; ///< battery info

    bool sfpro_; ///< are we a snapdragon pro board?
    
    bool tui_init_ = false; ///< indicates if the ncurses tui is initialized
    bool armed_ = false; ///< indicates if esc_interface is armed

    bool arm() { srvcall(true); }

    bool disarm() { srvcall(false); }

    bool srvcall(bool arm)
    {
        // true for arm, false for disarm
        std_srvs::SetBool srv;
        srv.request.data = arm;

        if (srv_arm_.call(srv)) {
            ROS_WARN_STREAM(srv.response.message);
            armed_ = srv.response.success;
        } else {
            if (tui_init_) mvprintw(0, 0, "Failed to call arm service!");
            else ROS_ERROR("Failed to call arm service!");
            return false;
        }

        return armed_;
    }

    bool isarmed()
    {
        // true for arm, false for disarm
        std_srvs::Trigger srv;

        if (srv_isarmed_.call(srv)) {
            armed_ = srv.response.success;
        } else {
            if (tui_init_) mvprintw(0, 0, "Failed to call is_armed service!");
            else ROS_ERROR("Failed to call is_armed service!");
            return false;
        }

        return armed_;
    }

    void waitForServices()
    {
        // wait for snap stack to advertise services
        srv_arm_.waitForExistence();
        srv_isarmed_.waitForExistence();

        armed_ = isarmed();
    }

    void initTUI()
    {
        initscr();
        clear();

        start_color();

        // tweak the default ncurses color definitions
        init_color(COLOR_RED, 800, 0, 0);
        init_color(COLOR_YELLOW, 800, 800, 0);
        init_color(COLOR_WHITE, 1000, 1000, 1000);

        init_pair(1, COLOR_RED, COLOR_BLACK); // id, fg, bg
        init_pair(2, COLOR_YELLOW, COLOR_BLACK); // id, fg, bg
        init_pair(3, COLOR_WHITE, COLOR_BLACK); // id, fg, bg

        curs_set(0); // invisible cursor
        noecho(); // don't echo what getch gets

        tui_init_ = true;
    }

    void closeTUI()
    {
        clear();
        endwin();
        tui_init_ = false;
    }

    void updateTUI(float cputemp=-1.0f)
    {
        static constexpr char armedmsg[] = "Armed";
        static constexpr char disarmedmsg[] = "Disarmed";

        // get terminal size
        int row, col;
        getmaxyx(stdscr, row, col);

        if (armed_) {
            bkgd(COLOR_PAIR(1));
            attron(A_BOLD); // bold text on
            mvprintw(row/2, (col-strlen(armedmsg))/2, "%s", armedmsg);
        } else {
            bkgd(COLOR_PAIR(2));
            attron(A_BOLD); // bold text on
            mvprintw(row/2, (col-strlen(disarmedmsg))/2, "%s", disarmedmsg);
        }

        if (cputemp > 0.0f) {
            attron(COLOR_PAIR(3));
            attroff(A_BOLD);
            mvprintw(0, 0, "CPU Temp: %.1f C", cputemp);
        }

        attron(COLOR_PAIR(3));
        attroff(A_BOLD);
        mvprintw(1, 0, "V: %.2f, I: %.2f", voltage_, current_);

        attron(COLOR_PAIR(3));
        attron(A_BOLD);
        mvprintw(0, col-strlen(name_.c_str()), "%s", name_.c_str());

        // write buffer to screen
        refresh();
    }

    float getCPUTemp()
    {
        std::string zonefile = "/sys/class/thermal/thermal_zone";
        zonefile += (sfpro_) ? "1" : "0";
        zonefile += "/temp";

        // open sysfs thermal file
        std::ifstream thermal(zonefile);
        if (!thermal.is_open()) return -1;

        // read temperature from file (integer, in Celsius, scaled)
        int itemp = -1;
        thermal >> itemp;
        thermal.close();

        // sfpro is scaled by 10x
        float scale = (sfpro_) ? 10.0f : 1.0f;

        // return as float with correct scale
        return static_cast<float>(itemp) / scale;
    }

    void battCb(const sensor_msgs::BatteryStateConstPtr& msg)
    {
        voltage_ = msg->voltage;
        current_ = msg->current;
    }

};

} // ns snap
} // ns acl

// ============================================================================
// ============================================================================

int main(int argc, char *argv[])
{
    // install sig handler
    struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sa.sa_handler = handle_sigint;
    sa.sa_flags = 0; // not SA_RESTART
    sigaction(SIGINT, &sa, NULL);

    ros::init(argc, argv, "esc_interface", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh;
    acl::snap::ESCInterfaceNode obj(nh);
    obj.spin();
    return 0;
}
