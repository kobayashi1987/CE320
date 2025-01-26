import customtkinter as ctk
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
        """
        Initializes the GUI for the NAILER: Code Analysis Tool.

        Parameters:
            root (CTk): The root window for the GUI.
        """

        # Set theme and appearance
        ctk.set_appearance_mode("dark")  # Set dark mode
        ctk.set_default_color_theme("blue")  # Set theme color

        # Configure root window
        self.root = root
        self.root.title("NAILER: Code Analysis Tool")
        self.root.geometry("1280x1100")

        # Split the UI into left and right frames
        self.left_frame = ctk.CTkFrame(root, width=400, height=600)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_propagate(False)  # Prevent resizing
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.right_frame = ctk.CTkFrame(root, width=800, height=600, fg_color="gray15")
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_propagate(False)  # Prevent resizing
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Initialize the list to store generated plot paths
        self.generated_plots = []

        # -------------------- Left Frame Widgets --------------------
        # Input Directory
        self.input_label = ctk.CTkLabel(self.left_frame, text="Input Directory:")
        self.input_label.pack(padx=10, pady=5, anchor="w")
        self.input_entry = ctk.CTkEntry(self.left_frame, width=300, placeholder_text="Select input folder")
        self.input_entry.pack(padx=10, pady=5)
        self.input_browse = ctk.CTkButton(self.left_frame, text="Browse", command=self.browse_input)
        self.input_browse.pack(padx=10, pady=5)

        # Output Directory
        self.output_label = ctk.CTkLabel(self.left_frame, text="Output Directory:")
        self.output_label.pack(padx=10, pady=5, anchor="w")
        self.output_entry = ctk.CTkEntry(self.left_frame, width=300, placeholder_text="Select output folder")
        self.output_entry.pack(padx=10, pady=5)
        self.output_browse = ctk.CTkButton(self.left_frame, text="Browse", command=self.browse_output)
        self.output_browse.pack(padx=10, pady=5)

        # Metrics Options
        self.metrics_label = ctk.CTkLabel(self.left_frame, text="Metrics:")
        self.metrics_label.pack(padx=10, pady=10, anchor="w")
        self.metrics_vars = {
            "Cosine": ctk.StringVar(value="1"),
            "Euclidean": ctk.StringVar(value="1"),
            "Pearson": ctk.StringVar(value="0"),
            "Jaccard": ctk.StringVar(value="0"),
        }
        for metric, var in self.metrics_vars.items():
            checkbox = ctk.CTkCheckBox(self.left_frame, text=metric, variable=var, onvalue="1", offvalue="0")
            checkbox.pack(padx=20, pady=5, anchor="w")

        # Clustering Options
        self.clusters_var = ctk.IntVar(value=2)
        # Label to display the current value of the slider
        self.clusters_value_label = ctk.CTkLabel(self.left_frame, text=f"Number of Clusters: {self.clusters_var.get()}")
        self.clusters_value_label.pack(padx=10, pady=5, anchor="w")
        # Slider for selecting the number of clusters
        self.clusters_spinbox = ctk.CTkSlider(
            self.left_frame,
            from_=2,
            to=10,
            number_of_steps=9,
            variable=self.clusters_var,
            command=self.update_cluster_label,  # Bind slider updates to update label dynamically
        )
        self.clusters_spinbox.pack(padx=20, pady=10)

        # Start and Clear Buttons
        self.start_button = ctk.CTkButton(self.left_frame, text="Start Analysis", command=self.start_analysis,
                                          width=150)
        self.start_button.pack(padx=10, pady=20)
        self.clear_button = ctk.CTkButton(self.left_frame, text="Clear Fields", command=self.clear_fields, width=150)
        self.clear_button.pack(padx=10, pady=10)

        # Status Display
        self.status_text = ctk.CTkTextbox(self.left_frame, height=150, width=300)
        self.status_text.pack(padx=10, pady=10)

        # Generated Plots
        self.plot_list_label = ctk.CTkLabel(self.left_frame, text="Generated Plots:")
        self.plot_list_label.pack(padx=10, pady=5, anchor="w")
        self.plot_list_frame = ctk.CTkScrollableFrame(self.left_frame, width=300, height=150)
        self.plot_list_frame.pack(padx=10, pady=10)

        # -------------------- Right Frame Widgets --------------------
        self.plot_canvas = None  # Will hold the current plot display
        self.plot_display_frame = ctk.CTkFrame(self.right_frame, width=780, height=580, fg_color="gray15")
        self.plot_display_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def browse_input(self):
        """
        Opens a directory browser dialog to select the input directory.
        """

        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, directory)

    def browse_output(self):
        """
        Opens a directory browser dialog to select the output directory.
        """
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, directory)

    def update_cluster_label(self, value):
        """
        Updates the label displaying the current number of clusters.
        """
        self.clusters_value_label.configure(text=f"Number of Clusters: {int(float(value))}")

    def update_status(self, message, clear=False):
        """
        Updates the status text area with a message.

        Parameters:
            message (str): The message to display.
            clear (bool): Whether to clear the existing messages before adding the new one.
        """

        if clear:
            self.status_text.configure(state="normal")
            self.status_text.delete(1.0, tk.END)

        self.status_text.configure(state="normal")
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.configure(state="disabled")
        self.status_text.see(tk.END)

    def clear_fields(self):
        """
        Clears all input fields, resets the metrics and clustering slider,
        and clears the status and plot list.
        """
        # Clear input and output fields
        self.input_entry.delete(0, tk.END)
        self.output_entry.delete(0, tk.END)

        # Reset metrics checkboxes to default
        for var in self.metrics_vars.values():
            var.set("0")
        self.metrics_vars["Cosine"].set("1")
        self.metrics_vars["Euclidean"].set("1")

        # Reset clustering slider to default
        self.clusters_var.set(2)
        self.update_cluster_label(self.clusters_var.get())  # Update the cluster label

        # Clear the generated plots list and plot buttons
        self.generated_plots = []
        for widget in self.plot_list_frame.winfo_children():  # Clear all widgets in the scrollable frame
            widget.destroy()

        # Clear the status text
        self.update_status("", clear=True)

    def display_plot(self, plot_path):
        """
        Displays a static plot image in the plot area.
        Parameters:
            plot_path (str): Path to the plot image file.
        """
        # Clear existing plot
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()

        # Create a new matplotlib figure
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        img = mpimg.imread(plot_path)
        ax.imshow(img)
        ax.axis("off")  # Hide the axis for cleaner display

        # Embed the plot in the plot display frame
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_display_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_analysis(self):
        """
          Starts the analysis process in a separate thread.
          Validates inputs, disables the Start button, and runs the analysis function.
        """

        input_dir = self.input_entry.get()
        output_dir = self.output_entry.get()
        metrics = [metric for metric, var in self.metrics_vars.items() if var.get() == "1"]
        clusters = self.clusters_var.get()

        # Validate path inputs
        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Invalid input directory.")
            return
        if not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Invalid output directory.")
            return

        # Disable the start button during processing
        self.start_button.configure(state="disabled")
        self.update_status(f"Starting analysis with metrics: {', '.join(metrics)} and {clusters} clusters...",
                           clear=True)

        # Run analysis in a separate thread to keep the GUI responsive
        def analysis_thread():
            try:
                # Run the analysis
                run_analysis(input_dir=input_dir, output_dir=output_dir, metrics=metrics, clusters=clusters)
                self.update_status("Analysis completed successfully.")

                # Populate the plot list with generated plot paths
                self.generated_plots = []  # Reset the generated plots
                for widget in self.plot_list_frame.winfo_children():  # Clear existing buttons
                    widget.destroy()

                plot_dir = os.path.join(output_dir, "visualizations")
                if os.path.exists(plot_dir):  # Check if the directory exists
                    for plot_file in os.listdir(plot_dir):
                        plot_path = os.path.join(plot_dir, plot_file)
                        self.generated_plots.append(plot_path)

                        # Create a button for each plot
                        button = ctk.CTkButton(
                            self.plot_list_frame,
                            text=os.path.basename(plot_file),
                            width=360,
                            command=lambda p=plot_path: self.display_plot(p),
                        )
                        button.pack(pady=2)

                if not self.generated_plots:
                    self.update_status("No plots generated.")
                    messagebox.showinfo("No Plots", "No plots were generated during analysis.")
                else:
                    self.update_status("Plots have been successfully loaded into the GUI.")

            except Exception as e:
                self.update_status(f"Error: {str(e)}")
                messagebox.showerror("Error", str(e))
            finally:
                self.start_button.configure(state="normal")

        threading.Thread(target=analysis_thread).start()


if __name__ == "__main__":
    root = ctk.CTk()
    app = NailerGUI(root)
    root.mainloop()
