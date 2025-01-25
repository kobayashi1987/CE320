import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

from matplotlib import image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from code_analysis import main as run_analysis


class NailerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NAILER: Code Analysis Tool")

        # Input Directory
        self.input_label = tk.Label(root, text="Input Directory:")
        self.input_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.input_entry = tk.Entry(root, width=40)
        self.input_entry.grid(row=0, column=1, padx=10, pady=5)
        self.input_browse = tk.Button(root, text="Browse", command=self.browse_input)
        self.input_browse.grid(row=0, column=2, padx=10, pady=5)

        # Output Directory
        self.output_label = tk.Label(root, text="Output Directory:")
        self.output_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.output_entry = tk.Entry(root, width=40)
        self.output_entry.grid(row=1, column=1, padx=10, pady=5)
        self.output_browse = tk.Button(root, text="Browse", command=self.browse_output)
        self.output_browse.grid(row=1, column=2, padx=10, pady=5)

        # Metrics Options
        self.metrics_label = tk.Label(root, text="Metrics:")
        self.metrics_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.metrics_var = tk.StringVar(value="cosine euclidean")
        self.metrics_entry = tk.Entry(root, width=40, textvariable=self.metrics_var)
        self.metrics_entry.grid(row=2, column=1, padx=10, pady=5)

        # Clustering Options
        self.clusters_label = tk.Label(root, text="Number of Clusters:")
        self.clusters_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.clusters_var = tk.IntVar(value=2)
        self.clusters_spinbox = tk.Spinbox(root, from_=1, to=10, textvariable=self.clusters_var, width=5)
        self.clusters_spinbox.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        # Start Button
        self.start_button = tk.Button(root, text="Start Analysis", command=self.start_analysis)
        self.start_button.grid(row=4, column=0, columnspan=3, pady=10)

        # Status Display
        self.status_text = tk.Text(root, height=10, width=60, state="disabled")
        self.status_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

        # List of Plots
        self.plot_list_label = tk.Label(root, text="Generated Plots:")
        self.plot_list_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.plot_listbox = tk.Listbox(root, height=5, width=40)
        self.plot_listbox.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        self.view_button = tk.Button(root, text="View Plot", command=self.view_selected_plot)
        self.view_button.grid(row=6, column=2, padx=10, pady=5)

        self.generated_plots = []  # List to store generated plot file paths

        # Plot Display Area
        self.plot_frame = tk.Frame(root, borderwidth=2, relief="sunken", width=600, height=400)
        self.plot_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.plot_canvas = None  # Will hold the current plot

    def browse_input(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, directory)

    def browse_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, directory)

    def update_status(self, message):
        self.status_text.config(state="normal")
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.config(state="disabled")
        self.status_text.see(tk.END)

    def display_plot(self, plot_path):
        """
        Displays a static plot image in the plot area.
        Parameters:
            plot_path (str): Path to the plot image file.
        """
        # Clear the existing plot
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()

        # Load the image and embed it into the GUI
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        img = mpimg.imread(plot_path)
        ax.imshow(img)
        ax.axis("off")  # Remove axis for cleaner display

        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def view_selected_plot(self):
        """
        Loads the selected plot and displays it in the GUI.
        """
        selected_index = self.plot_listbox.curselection()
        if selected_index:
            plot_path = self.generated_plots[selected_index[0]]
            self.display_plot(plot_path)
        else:
            messagebox.showwarning("No Selection", "Please select a plot to view.")

    def start_analysis(self):
        input_dir = self.input_entry.get()
        output_dir = self.output_entry.get()
        metrics = self.metrics_var.get().split()
        clusters = self.clusters_var.get()

        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Invalid input directory.")
            return
        if not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Invalid output directory.")
            return

        # Disable the start button during processing
        self.start_button.config(state="disabled")
        self.update_status("Starting analysis...")

        # Run analysis in a separate thread to keep the GUI responsive
        def analysis_thread():
            try:
                # Run the analysis
                run_analysis(input=input_dir, output=output_dir, metrics=metrics, clusters=clusters)
                self.update_status("Analysis completed successfully.")

                # Populate the plot list with generated plot paths
                plot_dir = os.path.join(output_dir, "visualizations")
                for plot_file in os.listdir(plot_dir):
                    plot_path = os.path.join(plot_dir, plot_file)
                    self.generated_plots.append(plot_path)
                    self.plot_listbox.insert(tk.END, os.path.basename(plot_file))

                messagebox.showinfo("Success", "Analysis completed successfully!")
            except Exception as e:
                self.update_status(f"Error: {str(e)}")
                messagebox.showerror("Error", str(e))
            finally:
                self.start_button.config(state="normal")

        threading.Thread(target=analysis_thread).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = NailerGUI(root)
    root.mainloop()