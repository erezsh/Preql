# FAQ

## Technical Help

### I'm new to Python. How do I install Preql?

First, you need to make sure Python is installed, at version 3.6 or above. You can get it from [https://www.python.org/downloads/](https://www.python.org/downloads/).

Then, open your command-line or shell, and write the following:

```sh
python -m pip install preql-lang --user
```

The Python executable might be called `python3`, or `python3.9` etc.

The `--user` switch ensures you won't need special permissions for the installation.


## Community and Support

### I think I found a bug. What do I do?

We'll do our best to solve bugs as quickly as possible.

If you found a bug in Preql, open an issue here: [https://github.com/erezsh/Preql/issues/new](https://github.com/erezsh/Preql/issues/new)

Include any information you can share. The best way is with a minimal code example that runs and demonstrates the error.


### Where can I ask questions?

You can ask any question here: [https://github.com/erezsh/Preql/discussions](https://github.com/erezsh/Preql/discussions)

### Contact Me

If you want to contact me privately, you may do so through email, at erezshin at gmail.com, or through [twitter](https://twitter.com/erezsh).

I am also available for paid support.

## License

### Can I use Preql in my project/company/product?

Preql is completely free for personal use, including internal use in commercial companies.

For use in projects and products, as a library, the license differentiates between two kinds of use:

1. Projects or products that use Preql internally, and don't expose the language to the user, may consider Preql as using the MIT license. That also applies to commercial projects and products.

2. Projects or products that intentionally expose the language to the user, as part of their interface.
Such use is only allowed for non-commercial projects, and then they must include the Preql license.

If you would like to embed the Preql language in your commercial project, and to benefit from its interface, contact us to buy a license.

### Why not dual license with GPL, AGPL, or other OSI-approved license?

In the history of open-source, GPL and AGPL were often used as a subtle strategy to disuade unfair commercial use. Most companies, and especially corporations, didn't want to share their own code, and so they had to buy a license if they wanted to use it.

Unfortunately, GPL and even AGPL don't fully protect software from exploitation by competitors, particularly cloud providers.

That is why many open-source projects, who were once AGPL, BSD or Apache 2.0, have decide to start using to their own license. Famous examples include Redis, Confluent, MongoDB, and Elastic.


### Is Preql open-source?

Preql's license is in line with some definitions of "open source", but does not fit the definition outlined by the OSI.

For practical purposes, Preql can be called "source available", or transparent-source.

