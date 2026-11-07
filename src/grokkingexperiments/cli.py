# https://typer.tiangolo.com/tutorial/package/#create-a-project
import typer

from grokkingexperiments.main import main
from grokkingexperiments.utils import load_config

# Show help message if no arg given
# https://typer.tiangolo.com/tutorial/commands/#show-the-help-message-if-no-command-is-given
app = typer.Typer(no_args_is_help=True)

@app.command()
def train(config_path: str = "config.toml"):
    print("!!! CLI IS WORKING AND CODES ARE RUNNING !!!")

    config = load_config(config_path)
    main(config)


@app.command()
def goodbye():
    print("Goodbye ")

if __name__ == "__main__":
    app()