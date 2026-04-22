import argparse
import http.server
import os
import socketserver
import tempfile
import threading
import time
import webbrowser

from pyinstrument import Profiler

import tests.utils as test_utils
import weather_model_graphs as wmg


def main():
    parser = argparse.ArgumentParser(
        description="Profile graph creation with pyinstrument."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=425,
        help="Size of grid (NxN points). Default is 425 (~180k points).",
    )
    parser.add_argument(
        "--archetype",
        type=str,
        default="keisler",
        choices=["keisler", "oskarsson_hierarchical", "graphcast"],
        help="Graph archetype to create.",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Print the profile to the console instead of opening a flamegraph in the browser.",
    )
    parser.add_argument(
        "--save-flamegraph",
        type=str,
        nargs="?",
        const="pyinstrument_profile.html",
        help="Save the HTML flamegraph to a file (default: pyinstrument_profile.html).",
    )

    args = parser.parse_args()

    print(f"Generating input coordinates for N={args.N} ({args.N**2} points)...")
    xy = test_utils.create_fake_xy(N=args.N)

    # Get the graph creation function dynamically based on the argument
    fn_name = f"create_{args.archetype}_graph"
    create_fn = getattr(wmg.create.archetype, fn_name)

    print(f"Starting pyinstrument profiling for {fn_name}...")
    profiler = Profiler(interval=0.001)  # 1ms precision

    # Profile the function
    profiler.start()
    t0 = time.time()
    graph = create_fn(coords=xy)
    t1 = time.time()
    profiler.stop()

    print(f"Graph creation finished in {t1 - t0:.2f} seconds.")
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

    if args.save_flamegraph:
        with open(args.save_flamegraph, "w") as f:
            f.write(profiler.output_html())
        print(f"Detailed report saved to '{args.save_flamegraph}'.")

    if args.console:
        print("\n--- Profile Output ---")
        print(profiler.output_text(unicode=True, color=True))
    elif not args.save_flamegraph:
        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = os.path.join(temp_dir, "index.html")
            with open(html_path, "w") as f:
                f.write(profiler.output_html())

            class Handler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=temp_dir, **kwargs)

                def log_message(self, format, *args):
                    pass  # suppress noisy server logs

            # Find a free port by binding to port 0
            with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
                port = httpd.server_address[1]
                url = f"http://127.0.0.1:{port}"
                print(f"\nServing flamegraph at {url}")
                print("Press Ctrl+C to shut down the server and exit.")

                # Open the browser in a separate thread so we can start serving immediately
                def open_browser():
                    time.sleep(0.5)
                    webbrowser.open(url)

                threading.Thread(target=open_browser, daemon=True).start()

                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nShutting down server.")


if __name__ == "__main__":
    main()
