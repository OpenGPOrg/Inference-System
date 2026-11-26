#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <string>

namespace opengp {
namespace server {

struct HttpRequest {
    std::string                        method;
    std::string                        path;
    std::map<std::string, std::string> headers; // lowercased header names
    std::string                        body;
};

class HttpConnection {
public:
    explicit HttpConnection(int client_fd) : fd_(client_fd) {}

    bool send_json(int status, const std::string & status_text, const std::string & json_body);
    bool send_cors_preflight();
    bool begin_sse();
    bool send_sse_data(const std::string & data);

private:
    bool write_all(const char * data, size_t len);

    int  fd_        = -1;
    bool responded_ = false;
};

using HttpHandler = std::function<void(const HttpRequest &, HttpConnection &)>;

// Single-process POSIX TCP HTTP/1.1 + SSE server (serializes GPU inference requests safely)
class HttpServer {
public:
    HttpServer(std::string host, int port);
    ~HttpServer();

    HttpServer(const HttpServer &) = delete;
    HttpServer & operator=(const HttpServer &) = delete;

    bool listen_and_serve(const HttpHandler & handler, std::atomic<bool> & stop_flag);
    int  port() const { return port_; }

private:
    std::string host_;
    int         port_      = 8080;
    int         listen_fd_ = -1;
};

} // namespace server
} // namespace opengp
