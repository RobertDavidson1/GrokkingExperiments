# https://typer.tiangolo.com/tutorial/package/#create-a-project
import typer

from grokkingexperiments.main import main

# Show help message if no arg given
# https://typer.tiangolo.com/tutorial/commands/#show-the-help-message-if-no-command-is-given
app = typer.Typer(no_args_is_help=True)

@app.command()
def train():
    main(97, "mod_n", 0.4, 64, 128, 20_000)


@app.command()
def goodbye():
    print("Goodbye ")