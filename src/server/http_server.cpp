#include "server/http_server.h"

#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace opengp {
namespace server {

namespace {

std::string to_lower_ascii(std::string s) {
    for (char & c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string trim_ascii(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
    size_t e = s.size();
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
    return s.substr(b, e - b);
}

const char * kCorsHeaders =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

bool decode_chunked(const std::string & raw, std::string & out) {
    out.clear();
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t line_end = raw.find("\r\n", pos);
        if (line_end == std::string::npos) return false;
        std::string size_str = raw.substr(pos, line_end - pos);
        size_t semi = size_str.find(';');
        if (semi != std::string::npos) size_str.resize(semi);
        char * endp = nullptr;
        unsigned long chunk_len = std::strtoul(size_str.c_str(), &endp, 16);
        if (endp == size_str.c_str()) return false;
        pos = line_end + 2;
        if (chunk_len == 0) return true;
        if (pos + chunk_len + 2 > raw.size()) return false;
        out.append(raw, pos, chunk_len);
        pos += chunk_len + 2;
    }
    return false;
}

ssize_t recv_with_stop(int fd, char * buf, size_t len, const std::atomic<bool> & stop_flag) {
    constexpr int kTotalTimeoutMs = 10000;
    int waited_ms = 0;
    while (!stop_flag.load() && waited_ms < kTotalTimeoutMs) {
        pollfd pfd{};
        pfd.fd     = fd;
        pfd.events = POLLIN;
        int pr = poll(&pfd, 1, 200);
        if (pr < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (pr == 0) {
            waited_ms += 200;
            continue;
        }
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            if (!(pfd.revents & POLLIN)) return 0;
        }
        return recv(fd, buf, len, 0);
    }
    return -1;
}

bool read_http_request(int fd, HttpRequest & req, const std::atomic<bool> & stop_flag) {
    constexpr size_t kMaxHeaderBytes = 64ULL * 1024ULL;
    constexpr size_t kMaxBodyBytes   = 16ULL * 1024ULL * 1024ULL;

    std::string buf;
    buf.reserve(4096);

    size_t header_end = std::string::npos;
    char   tmp[4096];

    while ((header_end = buf.find("\r\n\r\n")) == std::string::npos) {
        ssize_t n = recv_with_stop(fd, tmp, sizeof(tmp), stop_flag);
        if (n <= 0) return false;
        buf.append(tmp, static_cast<size_t>(n));
        if (buf.size() > kMaxHeaderBytes) return false;
    }

    const std::string header_block = buf.substr(0, header_end);
    const size_t      body_start   = header_end + 4;

    // Parse request line: METHOD /path HTTP/1.1
    size_t line_end = header_block.find("\r\n");
    std::string req_line = (line_end == std::string::npos) ? header_block : header_block.substr(0, line_end);
    size_t sp1 = req_line.find(' ');
    if (sp1 == std::string::npos) return false;
    size_t sp2 = req_line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return false;

    req.method = req_line.substr(0, sp1);
    req.path   = req_line.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t qmark = req.path.find('?');
    if (qmark != std::string::npos) {
        req.path.resize(qmark);
    }

    // Parse headers
    req.headers.clear();
    if (line_end != std::string::npos) {
        size_t cur = line_end + 2;
        while (cur < header_block.size()) {
            size_t next = header_block.find("\r\n", cur);
            std::string line = (next == std::string::npos)
                                   ? header_block.substr(cur)
                                   : header_block.substr(cur, next - cur);
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string k = to_lower_ascii(trim_ascii(line.substr(0, colon)));
                std::string v = trim_ascii(line.substr(colon + 1));
                req.headers[k] = v;
            }
            if (next == std::string::npos) break;
            cur = next + 2;
        }
    }

    std::string leftover = buf.substr(body_start);

    auto te_it = req.headers.find("transfer-encoding");
    if (te_it != req.headers.end() && to_lower_ascii(te_it->second).find("chunked") != std::string::npos) {
        while (leftover.find("\r\n0\r\n\r\n") == std::string::npos &&
               leftover.rfind("0\r\n\r\n", 0) != 0) {
            ssize_t n = recv_with_stop(fd, tmp, sizeof(tmp), stop_flag);
            if (n <= 0) break;
            leftover.append(tmp, static_cast<size_t>(n));
            if (leftover.size() > kMaxBodyBytes) return false;
        }
        return decode_chunked(leftover, req.body);
    }

    size_t content_len = 0;
    auto cl_it = req.headers.find("content-length");
    if (cl_it != req.headers.end()) {
        content_len = static_cast<size_t>(std::strtoull(cl_it->second.c_str(), nullptr, 10));
    }
    if (content_len > kMaxBodyBytes) return false;

    while (leftover.size() < content_len) {
        ssize_t n = recv_with_stop(fd, tmp, sizeof(tmp), stop_flag);
        if (n <= 0) return false;
        leftover.append(tmp, static_cast<size_t>(n));
    }
    req.body = leftover.substr(0, content_len);
    return true;
}

} // namespace

bool HttpConnection::write_all(const char * data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd_, data + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool HttpConnection::send_json(int status, const std::string & status_text, const std::string & json_body) {
    responded_ = true;
    std::string header =
        "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n" +
        std::string(kCorsHeaders) +
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(json_body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    return write_all(header.data(), header.size()) && write_all(json_body.data(), json_body.size());
}

bool HttpConnection::send_cors_preflight() {
    responded_ = true;
    std::string header =
        std::string("HTTP/1.1 204 No Content\r\n") +
        kCorsHeaders +
        "Content-Length: 0\r\n"
        "Connection: close\r\n\r\n";
    return write_all(header.data(), header.size());
}

bool HttpConnection::begin_sse() {
    responded_ = true;
    std::string header =
        std::string("HTTP/1.1 200 OK\r\n") +
        kCorsHeaders +
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "X-Accel-Buffering: no\r\n"
        "Connection: close\r\n\r\n";
    return write_all(header.data(), header.size());
}

bool HttpConnection::send_sse_data(const std::string & data) {
    if (!responded_) return false;
    std::string frame = "data: " + data + "\n\n";
    return write_all(frame.data(), frame.size());
}

HttpServer::HttpServer(std::string host, int port) : host_(std::move(host)), port_(port) {}

HttpServer::~HttpServer() {
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
}

bool HttpServer::listen_and_serve(const HttpHandler & handler, std::atomic<bool> & stop_flag) {
    std::signal(SIGPIPE, SIG_IGN);

    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        std::perror("socket");
        return false;
    }

    int yes = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(static_cast<uint16_t>(port_));
    if (host_ == "0.0.0.0" || host_.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) <= 0) {
        std::fprintf(stderr, "[-] HttpServer: Invalid bind host '%s'\n", host_.c_str());
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        std::perror("bind");
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (listen(listen_fd_, 32) < 0) {
        std::perror("listen");
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    while (!stop_flag.load()) {
        pollfd pfd{};
        pfd.fd     = listen_fd_;
        pfd.events = POLLIN;
        int pr = poll(&pfd, 1, 250);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pr == 0 || !(pfd.revents & POLLIN)) continue;

        int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            break;
        }

        int nodelay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

        HttpRequest req;
        HttpConnection conn(client_fd);
        if (read_http_request(client_fd, req, stop_flag)) {
            handler(req, conn);
        }
        close(client_fd);
    }

    close(listen_fd_);
    listen_fd_ = -1;
    return true;
}

} // namespace server
} // namespace opengp
